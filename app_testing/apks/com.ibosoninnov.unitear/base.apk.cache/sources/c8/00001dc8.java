package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class MatrixAnimFb extends Table {
    public static void addOps(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addSqtAnim(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(1, z, false);
    }

    public static int createMatrixAnimFb(FlatBufferBuilder flatBufferBuilder, int i, boolean z) {
        flatBufferBuilder.startObject(2);
        addOps(flatBufferBuilder, i);
        addSqtAnim(flatBufferBuilder, z);
        return endMatrixAnimFb(flatBufferBuilder);
    }

    public static int createOpsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int endMatrixAnimFb(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static MatrixAnimFb getRootAsMatrixAnimFb(ByteBuffer byteBuffer) {
        return getRootAsMatrixAnimFb(byteBuffer, new MatrixAnimFb());
    }

    public static void startMatrixAnimFb(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(2);
    }

    public static void startOpsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public MatrixAnimFb __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
        int i2 = i - byteBuffer.getInt(i);
        this.vtable_start = i2;
        this.vtable_size = this.bb.getShort(i2);
    }

    public MatrixOpFb ops(int i) {
        return ops(new MatrixOpFb(), i);
    }

    public int opsLength() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public boolean sqtAnim() {
        int __offset = __offset(6);
        return (__offset == 0 || this.bb.get(__offset + this.bb_pos) == 0) ? false : true;
    }

    public static MatrixAnimFb getRootAsMatrixAnimFb(ByteBuffer byteBuffer, MatrixAnimFb matrixAnimFb) {
        return matrixAnimFb.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public MatrixOpFb ops(MatrixOpFb matrixOpFb, int i) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return matrixOpFb.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }
}