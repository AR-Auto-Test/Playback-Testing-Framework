package com.google.ar.sceneform.lullmodel;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedInts;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class KeyVariantPairDef extends Table {
    public static void addHashKey(FlatBufferBuilder flatBufferBuilder, long j) {
        flatBufferBuilder.addInt(1, (int) j, 0);
    }

    public static void addKey(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addValue(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(3, i, 0);
    }

    public static void addValueType(FlatBufferBuilder flatBufferBuilder, byte b2) {
        flatBufferBuilder.addByte(2, b2, 0);
    }

    public static int createKeyVariantPairDef(FlatBufferBuilder flatBufferBuilder, int i, long j, byte b2, int i2) {
        flatBufferBuilder.startObject(4);
        addValue(flatBufferBuilder, i2);
        addHashKey(flatBufferBuilder, j);
        addKey(flatBufferBuilder, i);
        addValueType(flatBufferBuilder, b2);
        return endKeyVariantPairDef(flatBufferBuilder);
    }

    public static int endKeyVariantPairDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static KeyVariantPairDef getRootAsKeyVariantPairDef(ByteBuffer byteBuffer) {
        return getRootAsKeyVariantPairDef(byteBuffer, new KeyVariantPairDef());
    }

    public static void startKeyVariantPairDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(4);
    }

    public KeyVariantPairDef __assign(int i, ByteBuffer byteBuffer) {
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

    public long hashKey() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return this.bb.getInt(__offset + this.bb_pos) & UnsignedInts.INT_MASK;
        }
        return 0L;
    }

    public String key() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer keyAsByteBuffer() {
        return __vector_as_bytebuffer(4, 1);
    }

    public ByteBuffer keyInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 4, 1);
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

    public static KeyVariantPairDef getRootAsKeyVariantPairDef(ByteBuffer byteBuffer, KeyVariantPairDef keyVariantPairDef) {
        return keyVariantPairDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}