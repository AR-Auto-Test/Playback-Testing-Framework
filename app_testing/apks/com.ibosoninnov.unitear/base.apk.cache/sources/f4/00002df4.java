package org.opencv.core;

import java.util.Arrays;
import java.util.List;

/* loaded from: classes2.dex */
public class MatOfRotatedRect extends Mat {
    private static final int _channels = 5;
    private static final int _depth = 5;

    public MatOfRotatedRect() {
    }

    public static MatOfRotatedRect fromNativeAddr(long j) {
        return new MatOfRotatedRect(j);
    }

    public void alloc(int i) {
        if (i > 0) {
            super.create(i, 1, CvType.makeType(5, 5));
        }
    }

    public void fromArray(RotatedRect... rotatedRectArr) {
        if (rotatedRectArr == null || rotatedRectArr.length == 0) {
            return;
        }
        int length = rotatedRectArr.length;
        alloc(length);
        float[] fArr = new float[length * 5];
        for (int i = 0; i < length; i++) {
            RotatedRect rotatedRect = rotatedRectArr[i];
            int i2 = i * 5;
            Point point = rotatedRect.center;
            fArr[i2 + 0] = (float) point.x;
            fArr[i2 + 1] = (float) point.y;
            Size size = rotatedRect.size;
            fArr[i2 + 2] = (float) size.width;
            fArr[i2 + 3] = (float) size.height;
            fArr[i2 + 4] = (float) rotatedRect.angle;
        }
        put(0, 0, fArr);
    }

    public void fromList(List<RotatedRect> list) {
        fromArray((RotatedRect[]) list.toArray(new RotatedRect[0]));
    }

    public RotatedRect[] toArray() {
        int i = (int) total();
        RotatedRect[] rotatedRectArr = new RotatedRect[i];
        if (i == 0) {
            return rotatedRectArr;
        }
        float[] fArr = new float[5];
        for (int i2 = 0; i2 < i; i2++) {
            get(i2, 0, fArr);
            rotatedRectArr[i2] = new RotatedRect(new Point(fArr[0], fArr[1]), new Size(fArr[2], fArr[3]), fArr[4]);
        }
        return rotatedRectArr;
    }

    public List<RotatedRect> toList() {
        return Arrays.asList(toArray());
    }

    public MatOfRotatedRect(long j) {
        super(j);
        if (!empty() && checkVector(5, 5) < 0) {
            throw new IllegalArgumentException("Incompatible Mat");
        }
    }

    public MatOfRotatedRect(Mat mat) {
        super(mat, Range.all());
        if (!empty() && checkVector(5, 5) < 0) {
            throw new IllegalArgumentException("Incompatible Mat");
        }
    }

    public MatOfRotatedRect(RotatedRect... rotatedRectArr) {
        fromArray(rotatedRectArr);
    }
}