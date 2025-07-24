package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.ar.schemas.lull.Vec3;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class SuggestedCollisionShapeDef extends Table {
    public static void addCenter(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addStruct(1, i, 0);
    }

    public static void addSize(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addStruct(2, i, 0);
    }

    public static void addType(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addInt(0, i, 0);
    }

    public static int endSuggestedCollisionShapeDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static SuggestedCollisionShapeDef getRootAsSuggestedCollisionShapeDef(ByteBuffer byteBuffer) {
        return getRootAsSuggestedCollisionShapeDef(byteBuffer, new SuggestedCollisionShapeDef());
    }

    public static void startSuggestedCollisionShapeDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(3);
    }

    public SuggestedCollisionShapeDef __assign(int i, ByteBuffer byteBuffer) {
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

    public Vec3 center() {
        return center(new Vec3());
    }

    public Vec3 size() {
        return size(new Vec3());
    }

    public int type() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.getInt(__offset + this.bb_pos);
        }
        return 0;
    }

    public static SuggestedCollisionShapeDef getRootAsSuggestedCollisionShapeDef(ByteBuffer byteBuffer, SuggestedCollisionShapeDef suggestedCollisionShapeDef) {
        return suggestedCollisionShapeDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public Vec3 center(Vec3 vec3) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return vec3.__assign(__offset + this.bb_pos, this.bb);
        }
        return null;
    }

    public Vec3 size(Vec3 vec3) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return vec3.__assign(__offset + this.bb_pos, this.bb);
        }
        return null;
    }
}