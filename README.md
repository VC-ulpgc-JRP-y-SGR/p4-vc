# Práctica 4. Detección de caras

## Filtro de orejas y nariz de cerdito

Uno de los proyectos realizados para esta práctica es un filtro que, tras procesar y detectar un rostro, aplica unas imágenes de unas orejas y nariz  de cerdito en los lugares correspondientes del rostro.

### Código

Para poder aplicar el filtro, se ha desarrollado una clase de python la cual hemos llamado **PiggyFilter**.

###### Clase PiggyFilter

```
class PiggyFilter:
    def __init__(self, earl_str, earr_str, nose_str):
        self.earl_str = earl_str
        self.earr_str = earr_str
        self.nose_str = nose_str
```

* Constructor

    Se le pasa por parámetros las cadenas de texto con las direcciones de las imágenes que se usarán en el filtro.

```
    def _set_images(self, mode):
        earl_img = cv2.imread(self.earl_str, mode)
        earr_img = cv2.imread(self.earr_str, mode)
        nose_img = cv2.imread(self.nose_str, mode)

        self.earl_img = cv2.resize(earl_img, self.dim_ears)
        self.earr_img = cv2.resize(earr_img, self.dim_ears)
        self.nose_img = cv2.resize(nose_img, self.dim_nose)
```

* Método **_set_images**

    Este método privado se encarga de cargar las imágenes de las orejas izquierda y derecha, así como la nariz, y redimensionarlas según las dimensiones definidas. El parámetro *mode* se utiliza para especificar el modo de carga de la imagen (cv2.IMREAD_UNCHANGED indica que se deben cargar canales de transparencia si están presentes).

```
    def _set_images_pos(self, shape)
        self.earl_pos = (shape[17][0]-int(self.dim_ears[0]/2), shape[17][1]-int(self.dim_ears[1]))
        self.earr_pos = (shape[26][0]-int(self.dim_ears[0]/2), shape[26][1]-int(self.dim_ears[1]))
        self.nose_pos = (shape[33][0]-int(self.dim_nose[0]/2), shape[33][1]-int(self.dim_nose[1]))

```

* Método **_set_images_pos**

    Otra función privada que establece las posiciones de las imágenes de orejas y nariz en relación con la forma facial detectada. Utiliza puntos específicos de la forma facial (*shape*) para determinar las posiciones.

```
    def _set_dims(self, width):
        self.dim_ears = (int(width/2), int(width/2))
        self.dim_nose = (int(width/2), int(width/4))
```

* Método **_set_dims**

    Establece las dimensiones de las orejas y la nariz en función del ancho de la cara detectada.

```
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

* Método **_combine_images**

    Este método privado combina las imágenes de orejas y nariz con el marco de video. Utiliza máscaras para combinar las regiones de interés de las imágenes y el marco de video.

```
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

* Método **apply**

    Este método es el principal que aplica el filtro. Recibe un marco de video y los valores de detección facial (*values*). Luego, extrae la región de interés de la cara, calcula las posiciones y dimensiones de las imágenes de orejas y nariz, y finalmente, aplica el filtro llamando a **_combine_images** para combinar las imágenes con el marco de video.

