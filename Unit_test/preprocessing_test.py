import numpy as np
from Code.preprocessing import preprocess_frame, postprocess_frame

def test_preprocess_frame_output_shape():
    """Test que verifica que el frame preprocesado tenga las mismas dimensiones que el input"""
    input_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_frame = preprocess_frame(input_frame)
    assert input_frame.shape == processed_frame.shape, "El shape del frame no debe cambiar"

def test_preprocess_frame_output_type():
    """Test que verifica que el tipo de dato de salida sea correcto (uint8)"""
    input_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_frame = preprocess_frame(input_frame)
    assert processed_frame.dtype == np.uint8, "El tipo de dato debe ser uint8"

def test_preprocess_frame_non_empty_output():
    """Test que verifica que la salida no esté vacía"""
    input_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_frame = preprocess_frame(input_frame)
    assert processed_frame.size > 0, "El frame de salida no debe estar vacío"

def test_preprocess_frame_histogram_changed():
    """Test que verifica que la ecualización del histograma haya modificado el frame"""
    input_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_frame = preprocess_frame(input_frame)
    assert not np.array_equal(input_frame, processed_frame), "El frame debe ser modificado por el procesamiento"

def test_postprocess_frame_output_shape():
    """Test que verifica que el frame postprocesado tenga las dimensiones correctas (sin canal de color)"""
    input_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_frame = postprocess_frame(input_frame)
    assert processed_frame.shape == (480, 640), "El shape debe ser (height, width) sin canales"

def test_postprocess_frame_output_type():
    """Test que verifica que el tipo de dato de salida sea float32"""
    input_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_frame = postprocess_frame(input_frame)
    assert processed_frame.dtype == np.float32, "El tipo de dato debe ser float32"

def test_postprocess_frame_non_empty_output():
    """Test que verifica que la salida no esté vacía"""
    input_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_frame = postprocess_frame(input_frame)
    assert processed_frame.size > 0, "El frame de salida no debe estar vacío"

def test_postprocess_frame_values_range():
    """Test que verifica que los valores estén en el rango esperado (0-255)"""
    input_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_frame = postprocess_frame(input_frame)
    assert processed_frame.min() >= 0 and processed_frame.max() <= 255, "Los valores deben estar en el rango 0-255"
