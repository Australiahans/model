import keras_cv
from keras_cv.models import ViTTiny16,ViTB16,ViTL16
import random
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import EfficientNetB5,EfficientNetB6,EfficientNetV2L,EfficientNetV2M,EfficientNetV2S,ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
import numpy as np
import cv2
import os
from file_utils import read_json,save_json,cv_imread


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


model_type = "ViTB16"
NUM_CLASS = 102
batch_size = 32
model_folder = "ip102_vit_pretrain_new"
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.125),
        #layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        #layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


if model_type == "ViTTiny16":
    IMG_SIZE = 224
    ModelClass = ViTTiny16
elif model_type == "ViTB16":
    IMG_SIZE = 224
    ModelClass = ViTB16
elif model_type == "ViTL16":
    IMG_SIZE = 224
    ModelClass = ViTL16

img_h,img_w = IMG_SIZE,IMG_SIZE


def parse_with_opencv(img_path,label_num_str):
    im = cv_imread(img_path)[:,:,:3][:,:,::-1]
    im = np.array(cv2.resize(im, (img_h, img_w)), np.float32)
    label_i = int(label_num_str)
    label = np.array(np.eye(NUM_CLASS)[label_i],np.float32)
    return im, label


def construct_ip102_data_list():
    import pandas as pd

    def get_path_and_label(csv_path):
        df = pd.read_csv(csv_path)
        img_paths = [f"ip102_v1.1/images/{v}" for v in df['img']]
        labels = [str(int(v)) for v in df['label']]
        return img_paths,labels

    train_folder_list, train_label_list = get_path_and_label("ip102/train.csv")
    valid_folder_list, valid_label_list = get_path_and_label("ip102/val.csv")
    test_folder_list, test_label_list = get_path_and_label("ip102/test.csv")

    return train_folder_list, train_label_list, valid_folder_list, valid_label_list, test_folder_list, test_label_list



def get_all_data_paths():
    file_json_path = f'{model_folder}/ip102_data_paths.json'

    if not os.path.isfile(file_json_path):
        train_folder_list, train_label_list, valid_folder_list, valid_label_list, test_folder_list, test_label_list = construct_ip102_data_list()

        res_json = {'train_folder_list':train_folder_list,
                    'train_label_list':train_label_list,
                    'valid_folder_list':valid_folder_list,
                    'valid_label_list':valid_label_list,
                    'test_folder_list': test_folder_list,
                    'test_label_list': test_label_list
                    }

        save_json(file_json_path,res_json)

    else:
        rj = read_json(file_json_path)
        train_folder_list = rj['train_folder_list']
        train_label_list = rj['train_label_list']
        valid_folder_list = rj['valid_folder_list']
        valid_label_list = rj['valid_label_list']
        test_folder_list = rj['test_folder_list']
        test_label_list = rj['test_label_list']

    return train_folder_list,train_label_list,valid_folder_list,valid_label_list,test_folder_list,test_label_list



def lambda_func(x,y):
    a,b = tf.numpy_function(parse_with_opencv, [x,y], Tout=[tf.float32,tf.float32])
    a.set_shape([None,None,3])
    b.set_shape([NUM_CLASS])
    #a = img_augmentation(a)
    return a,b


def data_generator(img_folder_path_list, img_label_list):
    dataset = tf.data.Dataset.from_tensor_slices((img_folder_path_list,img_label_list))
    #dataset = dataset.skip(1000)
    dataset = dataset.shuffle(len(img_folder_path_list))
    dataset = dataset.repeat(10)
    dataset = dataset.map(lambda_func,num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size,drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def valid_data_generator(img_folder_path_list, img_label_list):
    dataset = tf.data.Dataset.from_tensor_slices((img_folder_path_list,img_label_list))
    #dataset = dataset.skip(1000)
    #dataset = dataset.shuffle(len(img_folder_path_list))
    dataset = dataset.map(lambda_func)
    dataset = dataset.batch(batch_size,drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def build_model(num_classes=NUM_CLASS):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    #inputs_aug = img_augmentation(inputs)

    #inputs_aug1 = layers.RandomRotation(factor=0.125,name="inputs_aug1")(inputs)
    #inputs_aug2 = layers.RandomFlip(name="inputs_aug2")(inputs_aug1)
    #x = img_augmentation(inputs)
    #x = tf.cast(inputs, tf.float32)
    #x = tf.keras.applications.mobilenet.preprocess_input(x)
    #x = tf.keras.applications.resnet.preprocess_input(x)
    vit = ModelClass(
        include_rescaling=True,
        include_top=False,
        name=model_type,
        weights="imagenet",
        input_tensor=inputs,
        pooling="token_pooling",
        activation=tf.keras.activations.gelu,
    )
    vit.trainable = False

    # Rebuild top
    #x = layers.GlobalAveragePooling2D(name="avg_pool")(vit.output)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dense(1024, activation="relu", name="last_dense")(x)
    x = vit.output
    top_dropout_rate = 0.1
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name=model_type)

    return model



def do_training(epochs=20,learning_rate=0.001,cw=None):
    # 对于增加的plantvillage数据集，改为在每次training时生成，这样random-的部分就会随机采样
    train_folder_list, train_label_list, valid_folder_list, valid_label_list,test_folder_list,test_label_list = get_all_data_paths()
    train_dataset = data_generator(train_folder_list, train_label_list)
    valid_dataset = valid_data_generator(valid_folder_list, valid_label_list)

    CKPT_filepath = f'{model_folder}/{model_type}_pretrain.h5'
    
    model = build_model(num_classes=NUM_CLASS)

    if os.path.isfile(CKPT_filepath):
        #model = tf.keras.models.load_model(CKPT_filepath)
    	model.load_weights(CKPT_filepath)
    	print('weights loaded.')
    #else:
    #    model = build_model(num_classes=NUM_CLASS)

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    #swa_opt = tfa.optimizers.SWA(optimizer, start_averaging=5,average_period=10, lr=0.005)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    step_per_epoch=len(train_folder_list)//batch_size

    checkpoint = tf.keras.callbacks.ModelCheckpoint(CKPT_filepath,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    mode='min')

    class_weight = {0: 0.14814, 1: 0.01508, 2: 0.03267, 3: 0.0316, 4: 0.04576, 5: 0.05614, 6: 0.10109,
                    7: 0.11083, 8: 0.09747, 9: 0.11042, 10: 0.11781, 11: 0.133}

    class_weight = {0: 0.14814, 1: 0.02508, 2: 0.04267, 3: 0.0416, 4: 0.05576, 5: 0.06614, 6: 0.10109,
                    7: 0.11083, 8: 0.09747, 9: 0.11042, 10: 0.11781, 11: 0.133}

    class_weight = {0: 0.9876, 1: 0.1508, 2: 0.3267, 3: 0.316, 4: 0.4576, 5: 0.5614, 6: 1.0109,
                    7: 0.8088, 8: 0.9747, 9: 1.1042, 10: 1.1781, 11: 1.33}

    #swa_callback = SWA(start_epoch=5, swa_epoch=10)

    #swa_callback = tfa.callbacks.AverageModelCheckpoint(filepath=CKPT_filepath.replace('.h5','_SWA.h5'),update_weights=True)

    if cw is not None:
        model.fit(train_dataset,steps_per_epoch=step_per_epoch, epochs=epochs,
                  validation_data=valid_dataset,
                  validation_steps=len(valid_folder_list)//batch_size,
                  callbacks=[checkpoint],
                  class_weight=cw
                  )
    else:
        model.fit(train_dataset,steps_per_epoch=step_per_epoch, epochs=epochs,
                  validation_data=valid_dataset,
                  validation_steps=len(valid_folder_list)//batch_size,
                  callbacks=[checkpoint]
                  )


    print("Training done!")


def evaluate_model():

    train_folder_list,train_label_list,valid_folder_list,valid_label_list,test_folder_list,test_label_list = get_all_data_paths()
    #train_dataset = data_generator(train_folder_list, train_label_list)
    test_dataset = valid_data_generator(test_folder_list,test_label_list)

    folder_list = test_folder_list
    dataset = test_dataset

    from tqdm import tqdm
    import pandas as pd

    CKPT_filepath = f'{model_folder}/{model_type}_pretrain.h5'
    print(CKPT_filepath)
    model = build_model(NUM_CLASS)
    model.load_weights(CKPT_filepath)
    # model = tf.keras.models.load_model(CKPT_filepath)
    # model = tf.keras.models.load_model(CKPT_filepath,custom_objects={'Addons>SWA':tfa.optimizers.SWA})
    print('model loaded!')
    model.summary()
    # dataset = dataset.skip(8)

    # model.load_weights('content/deeplab_model_UDD.h5')

    count_top1 = 0
    count_top3 = 0
    count_total = 0

    top1_heatmap = np.zeros((NUM_CLASS,NUM_CLASS))
    top3_heatmap = np.zeros((NUM_CLASS,NUM_CLASS))
    softmax_weight_heatmap = np.zeros((NUM_CLASS,NUM_CLASS))

    all_lines = []

    all_lines_per_class = []

    all_class_counts = []
    for i in range(NUM_CLASS):
        all_class_counts.append({"top1_tp": 0, "top1_fp": 0, "top3_tp": 0, "top3_fp": 0, "sample_count": 0})

    import matplotlib.pyplot as plt

    # for j in tqdm(range(len(folder_list) // batch_size)):
    for val in tqdm(dataset.as_numpy_iterator()):
        predsBatch = model.predict(np.expand_dims((val[0][0]), axis=0))
        # plt.imshow(np.array(np.round(np.array((val[0][0]),np.float32)*1),np.uint8))
        # plt.show()
        this_label = np.array(val[1][0])
        this_pred = predsBatch[0]
        this_label_i = np.argmax(this_label)
        this_pred_sort = np.argsort(this_pred)

        top1_heatmap[this_label_i][this_pred_sort[-1]] += 1
        softmax_weight_heatmap[this_label_i] += np.array(this_pred)
        top3_heatmap[this_label_i][this_pred_sort[-1]] += 1
        top3_heatmap[this_label_i][this_pred_sort[-2]] += 1
        top3_heatmap[this_label_i][this_pred_sort[-3]] += 1

        is_top1 = 0
        is_top3 = 0

        if this_label_i == this_pred_sort[-1]:
            count_top1 += 1
            is_top1 = 1
            all_class_counts[this_label_i]["top1_tp"] += 1
        else:
            all_class_counts[this_pred_sort[-1]]["top1_fp"] += 1

        if this_label_i in this_pred_sort[-3:]:
            count_top3 += 1
            is_top3 = 1
            all_class_counts[this_label_i]["top3_tp"] += 1
        else:
            all_class_counts[this_pred_sort[-3:][0]]["top3_fp"] += 1
            all_class_counts[this_pred_sort[-3:][1]]["top3_fp"] += 1
            all_class_counts[this_pred_sort[-3:][2]]["top3_fp"] += 1

        count_total += 1
        all_class_counts[this_label_i]["sample_count"] += 1
        all_lines.append([int(this_label_i), str([int(v) for v in this_pred_sort[-3:][::-1]]), is_top1, is_top3])

    # dataset = dataset.skip(batch_size)

    print('total:', count_total)
    print('top1:', count_top1 / count_total)
    print('top3:', count_top3 / count_total)

    for k in range(len(all_class_counts)):
        this_label_i = k
        this_top1_tp = all_class_counts[k]['top1_tp']
        this_top1_fp = all_class_counts[k]['top1_fp']
        this_top3_tp = all_class_counts[k]['top3_tp']
        this_top3_fp = all_class_counts[k]['top3_fp']
        this_counts = all_class_counts[k]['sample_count']
        all_lines_per_class.append(
            [this_label_i, this_counts, this_top1_tp, this_top1_fp, this_top3_tp, this_top3_fp])

    df = pd.DataFrame(all_lines_per_class)
    df.to_excel(f'ip102_{model_type}_pretrain_per_class.xlsx',
                header=['标签类别', '样本数', 'top1_tp', 'top1_fp', 'top3_tp', 'top3_fp'])

    dff = pd.DataFrame(all_lines)
    dff.to_excel(f'ip102_{model_type}_pretrain.xlsx', header=['标签类别', '预测类别', 'is_top1', 'is_top3'])

    heatmap_json = {
    "top1":top1_heatmap.tolist(),
    "top3":top3_heatmap.tolist(),
    "weight":softmax_weight_heatmap.tolist()
    }

    save_json(CKPT_filepath.replace('.h5','.json'),heatmap_json)

    plt.imshow(top1_heatmap)
    plt.title("top1")
    plt.colorbar()
    plt.show()

    plt.imshow(top3_heatmap)
    plt.title("top3")
    plt.colorbar()
    plt.show()

    plt.imshow(softmax_weight_heatmap)
    plt.title("prob ditribution")
    plt.colorbar()
    plt.show()


    pass


def visualize_pred_distribution():
    import matplotlib.pyplot as plt
    import seaborn as sns

    top1 = [[35.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 320.0, 25.0, 15.0, 9.0, 6.0, 0.0, 8.0, 1.0, 7.0, 4.0, 1.0], [0.0, 54.0, 81.0, 13.0, 9.0, 3.0, 0.0, 4.0, 9.0, 3.0, 5.0, 2.0], [1.0, 23.0, 13.0, 99.0, 13.0, 14.0, 10.0, 8.0, 2.0, 2.0, 0.0, 4.0], [1.0, 17.0, 8.0, 18.0, 56.0, 14.0, 7.0, 3.0, 0.0, 2.0, 0.0, 5.0], [1.0, 9.0, 7.0, 33.0, 6.0, 38.0, 2.0, 5.0, 1.0, 0.0, 0.0, 5.0], [0.0, 3.0, 7.0, 12.0, 5.0, 2.0, 26.0, 0.0, 0.0, 1.0, 0.0, 3.0], [0.0, 15.0, 1.0, 3.0, 2.0, 3.0, 0.0, 27.0, 0.0, 2.0, 0.0, 1.0], [0.0, 7.0, 13.0, 1.0, 1.0, 2.0, 0.0, 0.0, 34.0, 2.0, 0.0, 1.0], [0.0, 11.0, 4.0, 6.0, 3.0, 4.0, 1.0, 0.0, 2.0, 17.0, 5.0, 1.0], [1.0, 14.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 31.0, 0.0], [0.0, 9.0, 2.0, 4.0, 4.0, 6.0, 3.0, 1.0, 0.0, 1.0, 4.0, 11.0]]

    top3 = [[37.0, 23.0, 9.0, 8.0, 10.0, 8.0, 4.0, 0.0, 4.0, 9.0, 4.0, 4.0], [1.0, 373.0, 279.0, 174.0, 96.0, 57.0, 9.0, 58.0, 27.0, 69.0, 37.0, 11.0], [2.0, 146.0, 153.0, 60.0, 42.0, 29.0, 13.0, 23.0, 31.0, 22.0, 25.0, 3.0], [3.0, 89.0, 46.0, 166.0, 75.0, 91.0, 26.0, 18.0, 9.0, 14.0, 1.0, 29.0], [1.0, 61.0, 33.0, 78.0, 97.0, 44.0, 26.0, 11.0, 5.0, 16.0, 4.0, 17.0], [1.0, 38.0, 26.0, 84.0, 38.0, 80.0, 10.0, 11.0, 9.0, 14.0, 0.0, 10.0], [1.0, 13.0, 17.0, 39.0, 27.0, 18.0, 38.0, 5.0, 3.0, 3.0, 0.0, 13.0], [1.0, 36.0, 24.0, 30.0, 6.0, 13.0, 1.0, 38.0, 3.0, 7.0, 1.0, 2.0], [0.0, 39.0, 47.0, 9.0, 10.0, 8.0, 6.0, 1.0, 48.0, 7.0, 3.0, 5.0], [1.0, 26.0, 29.0, 17.0, 11.0, 14.0, 4.0, 4.0, 3.0, 38.0, 7.0, 8.0], [1.0, 37.0, 32.0, 3.0, 10.0, 7.0, 0.0, 2.0, 2.0, 11.0, 42.0, 6.0], [1.0, 18.0, 10.0, 25.0, 13.0, 18.0, 10.0, 3.0, 0.0, 6.0, 6.0, 25.0]]

    w = [[32.96330001577735, 2.3728602088271487, 0.7814102589998129, 0.9015653956967355, 0.6327002953667069, 0.7387894403367454, 0.5124485507938275, 0.06945762642044429, 0.34700001443251693, 0.4031539312743462, 0.05470049768914542, 0.22261299550615377], [0.6003423323109621, 256.3550758488127, 41.6197693781869, 25.371776110941937, 17.534480885256016, 12.973703973830197, 3.0870957852622496, 13.406185365340036, 4.007694590397989, 12.492328749058288, 5.921473821624318, 3.6300730167258024], [0.5753714350951817, 50.13016440952197, 64.8652930771932, 13.888116066820658, 12.207272582516453, 7.875962544138929, 2.9726508984980455, 5.798663513183783, 9.709195815543808, 5.745328890594465, 6.931564326219274, 2.3004169647333583], [1.128268846607706, 26.776802883512573, 15.029557295634731, 69.07540882122703, 19.50110856192623, 22.08452290511923, 9.942021070171904, 7.865311132318311, 3.548543707425779, 5.016631514685287, 0.9378299049309362, 8.09399321973342], [0.6139726462800164, 16.614462600351544, 9.394256744355516, 21.686930674128234, 44.59927726350725, 13.074609626200981, 7.875372898557657, 3.6219538718887634, 1.5242743661265195, 4.948337376699783, 1.2062380206224077, 5.840313690610856], [0.7783946358229898, 12.318242175271735, 8.132012598885922, 28.64438515668735, 9.352010002097813, 26.53541209793184, 4.149067139642284, 5.348828980386315, 2.5690200598619413, 4.298986892514222, 0.5073065613376286, 4.366332479574339], [0.2799430970545893, 3.7256694773823256, 6.195585368972388, 10.842333987122402, 6.752112096204655, 4.083825179666746, 20.34128342358963, 1.4503841809946607, 1.0667553099556244, 1.2250644595296762, 0.17653939183804823, 2.860504032938479], [0.2618699721788513, 13.171145091531798, 4.539823332685046, 6.571111902710982, 2.2464925781750935, 3.7165514535736293, 0.6142735115481628, 18.70676338818157, 0.5617897372776497, 2.2417040687141707, 0.26172148207604096, 1.1067531559601775], [0.008925232169573695, 6.6724426541914, 12.240637377683015, 2.1412943931154587, 1.5195817841313328, 2.2653114863595647, 0.7823031214338698, 0.48945642769740516, 32.25240161770489, 1.4169191148944265, 0.4086165172383778, 0.8021103434088559], [0.22674926437966025, 9.870799497002736, 7.169803097960539, 4.909791667800164, 3.573657005239511, 4.615419670706615, 1.5599421849028658, 1.006349058874548, 2.055387409640389, 12.336566710320767, 4.3442741967232905, 2.331260226325753], [0.5289185115152557, 9.901521366962697, 5.677451626863331, 1.192735763870587, 1.799149126658449, 1.5595307459516334, 0.24332690889829678, 0.412335294964123, 0.44049503799305967, 2.571544680162333, 25.997647964628413, 0.6753429980526562], [0.11674238453258656, 6.997010380378924, 2.5611342401462025, 6.230822190642357, 4.05626692972146, 5.357452464522794, 3.396800267692015, 1.3382179934924352, 0.3355048892190098, 2.0872685348695086, 2.641668157019012, 9.881111712791608]]

    top1 = np.array(top1)
    top3 = np.array(top3)
    w = np.array(w)

    print(np.sum(top1,axis=1))
    for i in range(len(top1)):
        top1[i] = top1[i]/np.sum(top1[i])
        top3[i] = top3[i]/np.sum(top3[i])
        w[i] = w[i]/np.sum(w[i])

    sns.heatmap(top1, annot=True, linewidths=.5)
    #plt.axis('equal')
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title("top1")
    plt.show()

    sns.heatmap(top3, annot=True, linewidths=.5)
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title("top3")
    plt.show()

    sns.heatmap(w, annot=True, linewidths=.5)
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title("prob ditribution")
    plt.show()

    # plt.imshow(top1,cmap='red')
    # plt.title("top1")
    # plt.xlabel("Prediction")
    # plt.ylabel("Ground Truth")
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(top3)
    # plt.title("top3")
    # plt.xlabel("Prediction")
    # plt.ylabel("Ground Truth")
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(w)
    # plt.title("prob ditribution")
    # plt.xlabel("Prediction")
    # plt.ylabel("Ground Truth")
    # plt.colorbar()
    # plt.show()

    pass


if __name__ == '__main__':
    #cww = {0: 0.9876, 1: 0.1508, 2: 0.3267, 3: 0.316, 4: 0.4576, 5: 0.5614, 6: 1.0109, 7: 0.8088, 8: 0.9747, 9: 1.1042, 10: 1.1781, 11: 1.33}

    # do_training(5,learning_rate=0.001,cw=cww)
    # do_training(5,learning_rate=0.001,cw=cww)
    # do_training(5,learning_rate=0.0001,cw=cww)
    # do_training(5,learning_rate=0.0001,cw=cww)
    # do_training(5,learning_rate=0.00001,cw=cww)
    # do_training(5,learning_rate=0.00001,cw=cww)
    # do_training(5,learning_rate=0.000003,cw=cww)
    # do_training(5,learning_rate=0.000003,cw=cww)

    #evaluate_model()

    do_training(10,learning_rate=0.001)
    do_training(10,learning_rate=0.0001)
    do_training(10,learning_rate=0.00001)

    pass



