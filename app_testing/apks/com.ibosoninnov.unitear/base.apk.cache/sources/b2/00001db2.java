package com.google.ar.schemas.lull;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class VariantArrayDefImpl extends Table {
    public static void addValue(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addValueType(FlatBufferBuilder flatBufferBuilder, byte b2) {
        flatBufferBuilder.addByte(0, b2, 0);
    }

    public static int createVariantArrayDefImpl(FlatBufferBuilder flatBufferBuilder, byte b2, int i) {
        flatBufferBuilder.startObject(2);
        addValue(flatBufferBuilder, i);
        addValueType(flatBufferBuilder, b2);
        return endVariantArrayDefImpl(flatBufferBuilder);
    }

    public static int endVariantArrayDefImpl(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static VariantArrayDefImpl getRootAsVariantArrayDefImpl(ByteBuffer byteBuffer) {
        return getRootAsVariantArrayDefImpl(byteBuffer, new VariantArrayDefImpl());
    }

    public static void startVariantArrayDefImpl(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(2);
    }

    public VariantArrayDefImpl __assign(int i, ByteBuffer byteBuffer) {
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

    public Table value(Table table) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __union(table, __offset);
        }
        return null;
    }

    public byte valueType() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.get(__offset + this.bb_pos);
        }
        return (byte) 0;
    }

    public static VariantArrayDefImpl getRootAsVariantArrayDefImpl(ByteBuffer byteBuffer, VariantArrayDefImpl variantArrayDefImpl) {
        return variantArrayDefImpl.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}