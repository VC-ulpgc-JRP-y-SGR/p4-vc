# Práctica 4. Detección de caras

## Filtro de orejas y nariz de cerdito

Uno de los proyectos realizados para esta práctica es un filtro que, tras procesar y detectar un rostro, aplica unas imágenes de unas orejas y nariz  de cerdito en los lugares correspondientes del rostro.

### Código

Para poder aplicar el filtro, se ha desarrollado una clase de python la cual hemos llamado **PiggyFilter**.

###### Clase PiggyFilter

* Constructor

```python
class PiggyFilter:
    def __init__(self, earl_str, earr_str, nose_str):
        self.earl_str = earl_str
        self.earr_str = earr_str
        self.nose_str = nose_str
```

    Se le pasa por parámetros las cadenas de texto con las direcciones de las imágenes que se usarán en el filtro.


* Método **_set_images**

```python
    def _set_images(self, mode):
        earl_img = cv2.imread(self.earl_str, mode)
        earr_img = cv2.imread(self.earr_str, mode)
        nose_img = cv2.imread(self.nose_str, mode)

        self.earl_img = cv2.resize(earl_img, self.dim_ears)
        self.earr_img = cv2.resize(earr_img, self.dim_ears)
        self.nose_img = cv2.resize(nose_img, self.dim_nose)
```

    Este método privado se encarga de cargar las imágenes de las orejas izquierda y derecha, así como la nariz, y redimensionarlas según las dimensiones definidas. El parámetro *mode* se utiliza para especificar el modo de carga de la imagen (cv2.IMREAD_UNCHANGED indica que se deben cargar canales de transparencia si están presentes).

* Método **_set_images_pos**

```python
    def _set_images_pos(self, shape)
        self.earl_pos = (shape[17][0]-int(self.dim_ears[0]/2), shape[17][1]-int(self.dim_ears[1]))
        self.earr_pos = (shape[26][0]-int(self.dim_ears[0]/2), shape[26][1]-int(self.dim_ears[1]))
        self.nose_pos = (shape[33][0]-int(self.dim_nose[0]/2), shape[33][1]-int(self.dim_nose[1]))
```

    Otra función privada que establece las posiciones de las imágenes de orejas y nariz en relación con la forma facial detectada. Utiliza puntos específicos de la forma facial (*shape*) para determinar las posiciones.

* Método **_set_dims**

```python
    def _set_dims(self, width):
        self.dim_ears = (int(width/2), int(width/2))
        self.dim_nose = (int(width/2), int(width/4))
```

    Establece las dimensiones de las orejas y la nariz en función del ancho de la cara detectada.

* Método **_combine_images**

```python
    def _combine_images(self, frame, image, image_pos, dim):
        frame_roi = frame[image_pos[1]:image_pos[1]+dim[1], image_pos[0]:image_pos[0]+dim[0]]
        image_roi = image[:, :, 0:3]

        mask = image[:, :, 3]
        inversed_mask = cv2.bitwise_not(mask)

        frame_masked = cv2.bitwise_and(frame_roi, frame_roi, mask=inversed_mask)
        image_masked = cv2.bitwise_and(image_roi, image_roi, mask=mask)

        fused = cv2.add(frame_masked, image_masked)
        frame[image_pos[1]:image_pos[1]+dim[1], image_pos[0]:image_pos[0]+dim[0]] = fused
        
        return frame
```

    Este método privado combina las imágenes de orejas y nariz con el marco de video. Utiliza máscaras para combinar las regiones de interés de las imágenes y el marco de video.

* Método **apply**

```python
    def apply(self, frame, values):
        face, eyes, shape = values
        [face_x, _, face_width, _] = face

        if face_x <= -1:
           return frame
         
        [leye_x, _, _, _] = eyes

        if leye_x <= -1: 
            return frame

        self._set_dims(width = face_width)
        self._set_images(mode = cv2.IMREAD_UNCHANGED)
        self._set_images_pos(shape = shape)

        frame = self._combine_images(frame, self.earl_img, self.earl_pos, self.dim_ears)
        frame = self._combine_images(frame, self.earr_img, self.earr_pos, self.dim_ears)
        frame = self._combine_images(frame, self.nose_img, self.nose_pos, self.dim_nose)

        return frame
```

    Este método es el principal que aplica el filtro. Recibe un marco de video y los valores de detección facial (*values*). Luego, extrae la región de interés de la cara, calcula las posiciones y dimensiones de las imágenes de orejas y nariz, y finalmente, aplica el filtro llamando a **_combine_images** para combinar las imágenes con el marco de video.

###### Captura de vídeo de la cámara

Para capturar un vídeo desde la cámara se ha desarrollado una función que simplifique el código.

```python
def capture_cam():
    vid = cv2.VideoCapture(0)
    cam_text = "Camera 0"

    if not vid.isOpened():
        vid = cv2.VideoCapture(1)
        if not vid.isOpened():
            vid = cv2.VideoCapture(0)
            if not vid.isOpened():
                print('Camera error')
                exit(0)
        else:
            cam_text = "Camera 1"

    print(cam_text)

    vid.set(3,640);
    vid.set(4,480);

    return vid
```

##### Detección del rostro y aplicación del filtro

Este fragmento de código es un script que utiliza las clases y funciones previamente definidas para aplicar un filtro "Piggy" a la transmisión de video de la cámara. Desglosamos el código:

* Captura de la cámara

```python
vid = capture_cam()
```

    Se llama a la función **capture_cam** para obtener un objeto de captura de video (vid).

* Inicialización de detectores y filtro

```python
FDet = FaceDetectors.FaceDetector()
piggyFilter = PiggyFilter('./images/left_ear.png', './images/right_ear.png', './images/nose.png')
```
    Se crea una instancia de la clase **FaceDetector** y una instancia de la clase **PiggyFilter**.

* Bucle principal para la captura de video

```python
while True:
    ret, frame = vid.read()
```

    Se inicia un bucle infinito que captura continuamente frames de video de la cámara.

* Detección de cara y ojos

```python
    # search face
    values = FDet.SingleFaceEyesDetection(frame, FDet.FaceDetectors[1], FDet.EyeDetectors[1])
```

    Se utiliza el objeto *FDet* (instancia de **FaceDetector**) para realizar la detección de cara y ojos en el frame actual. La función **SingleFaceEyesDetection** devuelve valores asociados a la cara y los ojos si se encuentran.

* Aplicación del filtro Piggy

```python
    if values is not None:
        frame = piggyFilter.apply(frame = frame, values = values)        
```

    Si se detecta una cara y ojos (*values* no es None), se aplica el filtro Piggy al frame utilizando el objeto *piggyFilter*.

* Mostrar el frame con el filtro aplicado

```python
    cv2.imshow('Cam', frame)
```

    Muestra el frame resultante con el filtro aplicado en una ventana titulada 'Cam'.

* Detener el bucle con la tecla Esc

```python
    # stop with esc key
    if cv2.waitKey(20) == 27:
        break
```

    El bucle se detiene si la tecla Esc es presionada.

* Liberación de la cámara y cierre de ventanas

```python
# close windows and release camera
vid.release()
cv2.destroyAllWindows()
```

    Al salir del bucle, se libera el objeto de captura de video y se cierran todas las ventanas.

### GIF ejemplo

![Ejemplo PiggyFilter](piggy-filter.gif)