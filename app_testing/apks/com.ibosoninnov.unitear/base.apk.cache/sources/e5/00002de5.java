package org.opencv.core;

import c.b.a.a.a;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/* loaded from: classes2.dex */
public class MatOfByte extends Mat {
    private static final int _channels = 1;
    private static final int _depth = 0;

    public MatOfByte() {
    }

    public static MatOfByte fromNativeAddr(long j) {
        return new MatOfByte(j);
    }

    public void alloc(int i) {
        if (i > 0) {
            super.create(i, 1, CvType.makeType(0, 1));
        }
    }

    public void fromArray(byte... bArr) {
        if (bArr == null || bArr.length == 0) {
            return;
        }
        alloc(bArr.length / 1);
        put(0, 0, bArr);
    }

    public void fromList(List<Byte> list) {
        if (list == null || list.size() == 0) {
            return;
        }
        Byte[] bArr = (Byte[]) list.toArray(new Byte[0]);
        byte[] bArr2 = new byte[bArr.length];
        for (int i = 0; i < bArr.length; i++) {
            bArr2[i] = bArr[i].byteValue();
        }
        fromArray(bArr2);
    }

    public byte[] toArray() {
        int checkVector = checkVector(1, 0);
        if (checkVector >= 0) {
            byte[] bArr = new byte[checkVector * 1];
            if (checkVector == 0) {
                return bArr;
            }
            get(0, 0, bArr);
            return bArr;
        }
        StringBuilder x = a.x("Native Mat has unexpected type or size: ");
        x.append(toString());
        throw new RuntimeException(x.toString());
    }

    public List<Byte> toList() {
        byte[] array = toArray();
        Byte[] bArr = new Byte[array.length];
        for (int i = 0; i < array.length; i++) {
            bArr[i] = Byte.valueOf(array[i]);
        }
        return Arrays.asList(bArr);
    }

    public MatOfByte(long j) {
        super(j);
        if (!empty() && checkVector(1, 0) < 0) {
            throw new IllegalArgumentException("Incompatible Mat");
        }
    }

    public MatOfByte(Mat mat) {
        super(mat, Range.all());
        if (!empty() && checkVector(1, 0) < 0) {
            throw new IllegalArgumentException("Incompatible Mat");
        }
    }

    public void fromArray(int i, int i2, byte... bArr) {
        if (i >= 0) {
            Objects.requireNonNull(bArr);
            if (i2 >= 0 && i2 + i <= bArr.length) {
                if (bArr.length == 0) {
                    return;
                }
                alloc(i2 / 1);
                put(0, 0, bArr, i, i2);
                return;
            }
            StringBuilder x = a.x("invalid 'length' parameter: ");
            x.append(Integer.toString(i2));
            throw new IllegalArgumentException(x.toString());
        }
        throw new IllegalArgumentException("offset < 0");
    }

    public MatOfByte(byte... bArr) {
        fromArray(bArr);
    }

    public MatOfByte(int i, int i2, byte... bArr) {
        fromArray(i, i2, bArr);
    }
}