package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class MatrixOpFb extends Table {
    public static void addId(FlatBufferBuilder flatBufferBuilder, byte b2) {
        flatBufferBuilder.addByte(0, b2, 0);
    }

    public static void addType(FlatBufferBuilder flatBufferBuilder, byte b2) {
        flatBufferBuilder.addByte(1, b2, 0);
    }

    public static void addValue(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(3, i, 0);
    }

    public static void addValueType(FlatBufferBuilder flatBufferBuilder, byte b2) {
        flatBufferBuilder.addByte(2, b2, 0);
    }

    public static int createMatrixOpFb(FlatBufferBuilder flatBufferBuilder, byte b2, byte b3, byte b4, int i) {
        flatBufferBuilder.startObject(4);
        addValue(flatBufferBuilder, i);
        addValueType(flatBufferBuilder, b4);
        addType(flatBufferBuilder, b3);
        addId(flatBufferBuilder, b2);
        return endMatrixOpFb(flatBufferBuilder);
    }

    public static int endMatrixOpFb(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static MatrixOpFb getRootAsMatrixOpFb(ByteBuffer byteBuffer) {
        return getRootAsMatrixOpFb(byteBuffer, new MatrixOpFb());
    }

    public static void startMatrixOpFb(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(4);
    }

    public MatrixOpFb __assign(int i, ByteBuffer byteBuffer) {
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

    public byte id() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.get(__offset + this.bb_pos);
        }
        return (byte) 0;
    }

    public byte type() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return this.bb.get(__offset + this.bb_pos);
        }
        return (byte) 0;
    }

    public Table value(Table table) {
        int __offset = __offset(10);
        if (__offset != 0) {
            return __union(table, __offset);
        }
        return null;
    }

    public byte valueType() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return this.bb.get(__offset + this.bb_pos);
        }
        return (byte) 0;
    }

    public static MatrixOpFb getRootAsMatrixOpFb(ByteBuffer byteBuffer, MatrixOpFb matrixOpFb) {
        return matrixOpFb.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}