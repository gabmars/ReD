from IPython.display import Image, display
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import tensorflow as tf
import numpy as np
import PIL.Image
import vgg16

vgg16.maybe_download()

def upscale(im, n):
    return im.resize((np.int32(im.width*n),np.int32(im.height*n)),PIL.Image.ANTIALIAS)
def downscale(im, n):
    return im.resize((np.int32(im.width/n),np.int32(im.height/n)),PIL.Image.ANTIALIAS)
    
    
def load_image(im, max_size=None):
    image=im
    if max_size is not None:
        # Вычисление соответствующего коэффициента масштабирования
        # для обеспечения максимальной высоты и ширины,
        # сохраняя пропорции
        factor = max_size / np.max(image.size)
    
        # Масштабирование размеров изображения 
        size = np.array(image.size) * factor

        # Преобразование к целочисленному формату
        # (PIL не принимет нецелочисленные форматы)
        size = size.astype(int)

        # Изменение размеров изображения
        image = image.resize(size, PIL.Image.LANCZOS)

    # Преобразование изображения в массив
    return np.float32(image)
    
def save_image(image, filename):
    # Проверка на принадлежность значения пикселя к интервалу [0, 255]
    image = np.clip(image, 0.0, 255.0)
    
    # Преобразование изображения в байтовый формат
    image = image.astype(np.uint8)
    
    # Сохранение изображения в формате jpeg
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')
        
def plot_image_big(image):
    # Проверка на принадлежность значения пикселя к интервалу [0, 255]
    image = np.clip(image, 0.0, 255.0)

    # Преобразование изображения в байтовый формат
    image = image.astype(np.uint8)

    # Отображение изображения преобразованного к формату PIL
    display(PIL.Image.fromarray(image))

def plot_images(content_image, style_image, mixed_image):
    # Создание рисунка с под-фигурами
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Регулировка размеров фигур
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Использование интерполяции для сглаживания пикселей
    smooth = True
    
    # Тип интерполяции
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Фигура для контент изображения
    # Значение пикселей нормализовано в диапозоне [0, 1]
    # деленное на 255
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    # Фигура для генерируемого изображения
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")

    # Фигура для стилизованного изображения
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")

    # Удаление подписей на координатных осях
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Отображение рисунка 
    plt.show()
    
def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))
    
def create_content_loss(session, model, content_image, layer_ids):
    """
    Функция вычисляющая потери для контент изображения
    
    Параметры
    session: открытая сессия tensorflow для запуска графа модели
    model: экземпляр класса VGG16
    content_image: контент изображение в виде массива
    layer_ids: список идентификаторов слоев для контент изображения
    """
    
    # Создание feed-dict - словаря элементов графа и значений
    feed_dict = model.create_feed_dict(image=content_image)

    # Предоставление ссылок для тензора заданного слоя
    layers = model.get_layer_tensors(layer_ids)

    # Вычисление выходных значений слоя, 
    # когда на вход подается контент изображение 
    values = session.run(layers, feed_dict=feed_dict)

    with model.graph.as_default():
        # Инициализация пустого списка значения функции потерь
        layer_losses = []
    
        for value, layer in zip(values, layers):
            # Эти значения, которые рассчитываются для этого слоя 
            # в модели при вводе контент изображения. 
            value_const = tf.constant(value)

            # Функция потерь для этого уровня - это средняя квадратичная ошибка 
            # между значениями слоя при вводе контент и генерируемого изображений.
            loss = mean_squared_error(layer, value_const)

            # Добавление функции потерь для этого слоя
            # в список функций потерь
            layer_losses.append(loss)

        # Комбинированная потеря для всех слоев - это среднее значение. 
        total_loss = tf.reduce_mean(layer_losses)
        
    return total_loss

def gram_matrix(tensor):
    shape = tensor.get_shape()
    
    # Получение значения количество каналов признаков для входного тензора
    num_channels = int(shape[3])

    # Приведение тензора к двумерной форме
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    
    # Вычисление Грам матрицы, как произведение двумерной матрицы
    # на саму себя, но транспонированную. Вычесление представляет собой
    # точное произведение каналов признаков
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

def create_style_loss(session, model, style_image, layer_ids):
    """
    Функция вычисляющая потери для стилизованного изображения
    
    Параметры
    session: открытая сессия tensorflow для запуска графа модели
    model: экземпляр класса VGG16
    content_image: контент изображение в виде массива
    layer_ids: список идентификаторов слоев для контент изображения
    """

    # Создание feed-dict - словаря элементов графа и значений
    feed_dict = model.create_feed_dict(image=style_image)

    # Предоставление ссылок для тензора заданного слоя
    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():
        # Создание операции tensorflow для вычисления
        # Грам матрицы для кажого слоя
        gram_layers = [gram_matrix(layer) for layer in layers]

        # Вычисление выходных значений слоя, 
        # когда на вход подается стилизованного изображение 
        values = session.run(gram_layers, feed_dict=feed_dict)

        # Инициализация пустого списка значения функции потерь
        layer_losses = []
    
        for value, gram_layer in zip(values, gram_layers):
            # Эти значения, которые рассчитываются для этого слоя 
            # в модели при вводе контент изображения. 
            value_const = tf.constant(value)

            # Функция потерь для этого уровня - это средняя квадратичная ошибка 
            # между значениями слоя при вводе контент и генерируемого изображений.
            loss = mean_squared_error(gram_layer, value_const)

            # Добавление функции потерь для этого слоя
            # в список функций потерь
            layer_losses.append(loss)

        # Комбинированная потеря для всех слоев - это среднее значение. 
        total_loss = tf.reduce_mean(layer_losses)
        
    return total_loss
    
def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
           tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))

    return loss
    
def style_transfer(content_image, style_image,
                   content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0):
    """
    Используется градиентный спуск, чтобы найти изображение
    минимизирующее функцию потреь контенти стиль изображения.
    Это должно привести к смешанному изображению, которое напоминает контуры 
    содержимого-изображения и состоит из цвета и текстур стиль изображения.
    
    Параметры
    content_image: Трехмерный массив содержащий контент изображение
    style_image: Трехмерный массив содержащий стиль изображение
    content_layer_ids: Спиок идентификаторов слоев для контента
    style_layer_ids: Спиок идентификаторов слоев для стиля
    weight_content: Вес функции потерь контента
    weight_style: Вес функции потерь стиля
    weight_denoise: Вес функции потерь шума
    num_iterations: Количество итераций
    step_size: Шаг для градиента на каждой итерации
    """

    # Создание экземпляра VGG16
    model = vgg16.VGG16()
    # Создание интерактивной сессии Tensorflow
    session = tf.InteractiveSession(graph=model.graph)

    # Вывод названий слоев контента
    print("Content layers:")
    print(model.get_layer_names(content_layer_ids))
    print()

    # Вывод названий слоев стиля
    print("Style layers:")
    print(model.get_layer_names(style_layer_ids))
    print()

    # Вычисление функции потерь контент изображения
    loss_content = create_content_loss(session=session,
                                       model=model,
                                       content_image=content_image,
                                       layer_ids=content_layer_ids)

    # Вычисление функции потерь стиль изображения
    loss_style = create_style_loss(session=session,
                                   model=model,
                                   style_image=style_image,
                                   layer_ids=style_layer_ids)    

    # Вычисление функции потерь шума
    loss_denoise = create_denoise_loss(model)

    # Создание переменных Tensorflow для отрегулированных функций потерь
    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    # Инизиализация отрегулированных занчений функций потерь
    session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])

    # Создание операции Tensorflow для обновления отрегулированных значений
    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    # Вычисление весов функции потерь для минимизации
    loss_combined = weight_content * adj_content * loss_content + \
                    weight_style * adj_style * loss_style + \
                    weight_denoise * adj_denoise * loss_denoise

    # Вычисление градиента для комбинированного значения функций потреь
    gradient = tf.gradients(loss_combined, model.input)
    
    # Список тензоров для запуска при оптимизации 
    run_list = [gradient, update_adj_content, update_adj_style, \
                update_adj_denoise]

    # Создание изображения для смешивания, как случайного шума
    mixed_image = np.random.rand(*content_image.shape) + 128
  
    for i in range(num_iterations):
        # Создание feed-dict - словаря элементов графа и значений
        feed_dict = model.create_feed_dict(image=mixed_image)

        # Вычисление значений градиента и отрегулированных значений
        grad, adj_content_val, adj_style_val, adj_denoise_val = session.run(run_list, feed_dict=feed_dict)

        # Уменьшение размерности градиента
        grad = np.squeeze(grad)

        # Масштабирование в соответствии с параметром step size
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Обновление изображения с помощью градиента
        mixed_image -= grad * step_size_scaled

        # Проверка на принадлежность значения пикселя к интервалу [0, 255]
        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        # Индикатор прогресса 
        print(". ", end="")

        # Отображение весов и изображений каждые 10 итераций
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)

            msg = "Adjust weights for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

            plot_images(content_image=content_image,
                        style_image=style_image,
                        mixed_image=mixed_image)
            
    print()
    print("Final image:")
    mixed_image = np.clip(mixed_image, 0.0, 255.0)
    mixed_image = mixed_image.astype(np.uint8)
    mixed_image=PIL.Image.fromarray(mixed_image)
#    mixed_image=upscale(mixed_image,4)
    imshow(mixed_image)
#    plot_image_big(mixed_image)

    # Закрытие сессии Tensorflow
    session.close()
    
    # Возвращение итогового изображения
    return mixed_image

content = PIL.Image.open('content/cat.jpg')
#content = downscale(content, 4)
content_image = load_image(content, max_size=300)
plot_image_big(content_image)


style = PIL.Image.open('style/a2.jpg')
#style = downscale(style, 4)
style_image = load_image(style, max_size=300)
plot_image_big(style_image)


content_layer_ids = [1]

style_layer_ids = [0,2,4,7,10]
#style_layer_ids = list(range(13))

img = style_transfer(content_image=content_image, 
                     style_image=style_image,
                     content_layer_ids=content_layer_ids,
                     style_layer_ids=style_layer_ids,
                     weight_content=1.0,
                     weight_style=3.0,
                     weight_denoise=0.2,
                     num_iterations=250,
                     step_size=8.0)
