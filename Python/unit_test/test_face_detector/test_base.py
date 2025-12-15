# test_base.py
import unittest
from abc import ABCMeta
from src.face_detector.base import BaseFaceDetector

class TestBaseFaceDetector(unittest.TestCase):
    """Unit tests for the base class BaseFaceDetector."""

    def test_class_is_abstract(self):
        """Test: Verify that BaseFaceDetector is an abstract class."""
        # Check that it has the __abstractmethods__ attribute (from abc.ABC)
        self.assertTrue(hasattr(BaseFaceDetector, '__abstractmethods__'))
        self.assertIsInstance(BaseFaceDetector, ABCMeta)

    def test_cannot_instantiate_directly(self):
        """Test: BaseFaceDetector cannot be instantiated directly."""
        with self.assertRaises(TypeError) as context:
            detector = BaseFaceDetector()
        
        # Verify the error message indicates abstract methods
        error_msg = str(context.exception)
        self.assertIn("abstract", error_msg.lower())
        self.assertIn("method", error_msg.lower())

    def test_has_required_abstract_methods(self):
        """Test: Verify that the class has the required abstract methods."""
        # Check that methods exist as abstract methods
        abstract_methods = BaseFaceDetector.__abstractmethods__
        
        self.assertIn('detect', abstract_methods)
        self.assertIn('close', abstract_methods)
        
        # Verify that the methods are defined in the class
        self.assertTrue(hasattr(BaseFaceDetector, 'detect'))
        self.assertTrue(hasattr(BaseFaceDetector, 'close'))

    def test_detect_method_signature(self):
        """Test: Verify the signature of the detect method."""
        # Get the method annotation (if exists)
        detect_method = BaseFaceDetector.detect
        
        # Verify it has documentation
        self.assertIsNotNone(detect_method.__doc__)
        
        # Verify the documentation mentions expected parameters and return values
        docstring = detect_method.__doc__
        self.assertIn("frame", docstring.lower())
        self.assertIn("numpy.ndarray", docstring)
        self.assertIn("tuple", docstring)
        self.assertIn("none", docstring.lower())
        self.assertIn("x, y, w, h", docstring)

    def test_close_method_signature(self):
        """Test: Verify the signature of the close method."""
        # Get the method annotation
        close_method = BaseFaceDetector.close
        
        # Verify it has documentation
        self.assertIsNotNone(close_method.__doc__)
        
        # Verify the documentation mentions resource release
        docstring = close_method.__doc__
        self.assertIn("release", docstring.lower())
        self.assertIn("resources", docstring.lower())

    def test_concrete_implementation_can_be_instantiated(self):
        """Test: A concrete implementation can be instantiated."""
        # Create a minimal concrete implementation
        class ConcreteDetector(BaseFaceDetector):
            def detect(self, frame):
                """Concrete implementation of detect."""
                return (10, 20, 30, 40)
            
            def close(self):
                """Concrete implementation of close."""
                pass
        
        # Verify it can be instantiated
        detector = ConcreteDetector()
        self.assertIsInstance(detector, BaseFaceDetector)
        self.assertIsInstance(detector, ConcreteDetector)

    def test_concrete_implementation_must_implement_all_abstract_methods(self):
        """Test: A concrete implementation must implement all abstract methods."""
        # Try to create an incomplete implementation
        class IncompleteDetector(BaseFaceDetector):
            def detect(self, frame):
                return None
            # Missing close() implementation
        
        # Verify it cannot be instantiated
        with self.assertRaises(TypeError) as context:
            detector = IncompleteDetector()
        
        error_msg = str(context.exception)
        self.assertIn("abstract", error_msg.lower())

    def test_concrete_implementation_methods_work(self):
        """Test: Methods of a concrete implementation work correctly."""
        # Create a concrete implementation with specific logic
        class TestDetector(BaseFaceDetector):
            def __init__(self):
                self.call_count = 0
            
            def detect(self, frame):
                """Returns coordinates based on call count."""
                self.call_count += 1
                return (self.call_count * 10, 20, 30, 40)
            
            def close(self):
                """Resets the count."""
                self.call_count = 0
        
        # Instantiate and test
        detector = TestDetector()
        
        # Test detect
        result1 = detector.detect(None)
        self.assertEqual(result1, (10, 20, 30, 40))
        
        result2 = detector.detect(None)
        self.assertEqual(result2, (20, 20, 30, 40))
        
        # Test close
        detector.close()
        result3 = detector.detect(None)
        self.assertEqual(result3, (10, 20, 30, 40))

    def test_multiple_inheritance_with_other_classes(self):
        """Test: BaseFaceDetector can be part of multiple inheritance."""
        class OtherClass:
            def other_method(self):
                return "other"
        
        class MultiInheritanceDetector(BaseFaceDetector, OtherClass):
            def detect(self, frame):
                return (0, 0, 100, 100)
            
            def close(self):
                pass
        
        # Verify it works with multiple inheritance
        detector = MultiInheritanceDetector()
        self.assertIsInstance(detector, BaseFaceDetector)
        self.assertIsInstance(detector, OtherClass)
        self.assertEqual(detector.other_method(), "other")
        self.assertEqual(detector.detect(None), (0, 0, 100, 100))

    def test_subclass_can_add_new_methods(self):
        """Test: Subclasses can add additional methods."""
        class ExtendedDetector(BaseFaceDetector):
            def __init__(self):
                self.initialized = True
            
            def detect(self, frame):
                return (50, 60, 70, 80)
            
            def close(self):
                self.initialized = False
            
            def new_method(self):
                """Additional method specific to the subclass."""
                return "extra functionality"
        
        detector = ExtendedDetector()
        self.assertTrue(detector.initialized)
        self.assertEqual(detector.new_method(), "extra functionality")
        self.assertEqual(detector.detect(None), (50, 60, 70, 80))
        
        detector.close()
        self.assertFalse(detector.initialized)

    def test_detect_can_return_none(self):
        """Test: The detect method can return None according to the documentation."""
        class NoneReturningDetector(BaseFaceDetector):
            def detect(self, frame):
                # Implementation that can return None
                return None
            
            def close(self):
                pass
        
        detector = NoneReturningDetector()
        result = detector.detect(None)
        self.assertIsNone(result)

    def test_class_documentation(self):
        """Test: Verify that the class has proper documentation."""
        self.assertIsNotNone(BaseFaceDetector.__doc__)
        
        docstring = BaseFaceDetector.__doc__
        # Verify the documentation mentions key concepts
        self.assertIn("Abstract base class", docstring)
        self.assertIn("face detection", docstring.lower())
        self.assertIn("interface", docstring.lower())
        self.assertIn("detect", docstring.lower())
        self.assertIn("close", docstring.lower())