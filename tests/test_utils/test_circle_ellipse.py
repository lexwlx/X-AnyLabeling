import unittest

from PyQt5 import QtCore

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.shape import shape_to_mask


class TestCircleEllipse(unittest.TestCase):

    def test_shape_to_mask_circle_legacy_points(self):
        mask = shape_to_mask(
            img_shape=(100, 100, 3),
            points=[[50, 50], [60, 50]],
            shape_type="circle",
        )
        self.assertTrue(mask[50, 50])
        self.assertTrue(mask[50, 60])
        self.assertFalse(mask[50, 75])

    def test_shape_to_mask_ellipse_5_points(self):
        points = [
            [50, 50],  # center
            [70, 50],  # right
            [50, 60],  # bottom
            [30, 50],  # left
            [50, 40],  # top
        ]
        mask = shape_to_mask(
            img_shape=(100, 100, 3),
            points=points,
            shape_type="circle",
        )
        self.assertTrue(mask[50, 50])
        self.assertTrue(mask[50, 70])
        self.assertTrue(mask[50, 30])
        self.assertFalse(mask[80, 50])

    def test_convert_circle_to_ellipse_handles(self):
        shape = Shape(shape_type="circle")
        shape.points = [QtCore.QPointF(10, 10), QtCore.QPointF(13, 14)]
        self.assertTrue(shape.convert_circle_to_ellipse())

        self.assertEqual(len(shape.points), 5)
        self.assertEqual((shape.points[1].x(), shape.points[1].y()), (15.0, 10.0))
        self.assertEqual((shape.points[2].x(), shape.points[2].y()), (10.0, 15.0))
        self.assertEqual((shape.points[3].x(), shape.points[3].y()), (5.0, 10.0))
        self.assertEqual((shape.points[4].x(), shape.points[4].y()), (10.0, 5.0))

    def test_move_ellipse_handle_updates_opposite(self):
        shape = Shape(shape_type="circle")
        shape.points = [
            QtCore.QPointF(50, 50),
            QtCore.QPointF(70, 50),
            QtCore.QPointF(50, 60),
            QtCore.QPointF(30, 50),
            QtCore.QPointF(50, 40),
        ]

        # Drag right handle 5px to the right (y offset should be ignored).
        shape.move_vertex_by(1, QtCore.QPointF(5, 123))

        self.assertEqual((shape.points[1].x(), shape.points[1].y()), (75.0, 50.0))
        self.assertEqual((shape.points[3].x(), shape.points[3].y()), (25.0, 50.0))
        # Vertical handles should stay centered on x.
        self.assertEqual(shape.points[2].x(), 50.0)
        self.assertEqual(shape.points[4].x(), 50.0)

    def test_shape_to_mask_rectangle_with_4_points(self):
        mask = shape_to_mask(
            img_shape=(100, 100, 3),
            points=[[10, 10], [30, 10], [30, 30], [10, 30]],
            shape_type="rectangle",
        )
        self.assertTrue(mask[20, 20])
        self.assertFalse(mask[40, 40])

    def test_shape_to_mask_out_of_bounds_ellipse_is_clipped(self):
        # Ellipse extends beyond the right image border; mask should still be valid
        # and only include pixels inside the image canvas.
        mask = shape_to_mask(
            img_shape=(100, 100, 3),
            points=[
                [95, 50],  # center
                [120, 50],  # right (outside image)
                [95, 70],  # bottom
                [70, 50],  # left
                [95, 30],  # top
            ],
            shape_type="circle",
        )
        self.assertTrue(mask[50, 99])
        self.assertFalse(mask[5, 5])
