package com.google.ar.schemas.lull;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class SubmeshAabb extends Table {
    public static void addMaxPosition(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addStruct(1, i, 0);
    }

    public static void addMinPosition(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addStruct(0, i, 0);
    }

    public static int endSubmeshAabb(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static SubmeshAabb getRootAsSubmeshAabb(ByteBuffer byteBuffer) {
        return getRootAsSubmeshAabb(byteBuffer, new SubmeshAabb());
    }

    public static void startSubmeshAabb(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(2);
    }

    public SubmeshAabb __assign(int i, ByteBuffer byteBuffer) {
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

    public Vec3 maxPosition() {
        return maxPosition(new Vec3());
    }

    public Vec3 minPosition() {
        return minPosition(new Vec3());
    }

    public static SubmeshAabb getRootAsSubmeshAabb(ByteBuffer byteBuffer, SubmeshAabb submeshAabb) {
        return submeshAabb.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public Vec3 maxPosition(Vec3 vec3) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return vec3.__assign(__offset + this.bb_pos, this.bb);
        }
        return null;
    }

    public Vec3 minPosition(Vec3 vec3) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return vec3.__assign(__offset + this.bb_pos, this.bb);
        }
        return null;
    }
}