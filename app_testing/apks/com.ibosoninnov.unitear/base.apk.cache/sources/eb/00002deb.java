package org.opencv.core;

import c.b.a.a.a;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes2.dex */
public class MatOfInt extends Mat {
    private static final int _channels = 1;
    private static final int _depth = 4;

    public MatOfInt() {
    }

    public static MatOfInt fromNativeAddr(long j) {
        return new MatOfInt(j);
    }

    public void alloc(int i) {
        if (i > 0) {
            super.create(i, 1, CvType.makeType(4, 1));
        }
    }

    public void fromArray(int... iArr) {
        if (iArr == null || iArr.length == 0) {
            return;
        }
        alloc(iArr.length / 1);
        put(0, 0, iArr);
    }

    public void fromList(List<Integer> list) {
        if (list == null || list.size() == 0) {
            return;
        }
        Integer[] numArr = (Integer[]) list.toArray(new Integer[0]);
        int[] iArr = new int[numArr.length];
        for (int i = 0; i < numArr.length; i++) {
            iArr[i] = numArr[i].intValue();
        }
        fromArray(iArr);
    }

    public int[] toArray() {
        int checkVector = checkVector(1, 4);
        if (checkVector >= 0) {
            int[] iArr = new int[checkVector * 1];
            if (checkVector == 0) {
                return iArr;
            }
            get(0, 0, iArr);
            return iArr;
        }
        StringBuilder x = a.x("Native Mat has unexpected type or size: ");
        x.append(toString());
        throw new RuntimeException(x.toString());
    }

    public List<Integer> toList() {
        int[] array = toArray();
        Integer[] numArr = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            numArr[i] = Integer.valueOf(array[i]);
        }
        return Arrays.asList(numArr);
    }

    public MatOfInt(long j) {
        super(j);
        if (!empty() && checkVector(1, 4) < 0) {
            throw new IllegalArgumentException("Incompatible Mat");
        }
    }

    public MatOfInt(Mat mat) {
        super(mat, Range.all());
        if (!empty() && checkVector(1, 4) < 0) {
            throw new IllegalArgumentException("Incompatible Mat");
        }
    }

    public MatOfInt(int... iArr) {
        fromArray(iArr);
    }
}