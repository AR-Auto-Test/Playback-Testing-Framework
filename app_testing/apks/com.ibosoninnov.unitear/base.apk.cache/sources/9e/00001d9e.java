package com.google.ar.schemas.lull;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ModelDef extends Table {
    public static void addBoundingBox(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addStruct(3, i, 0);
    }

    public static void addLods(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addSkeleton(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static void addTextures(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(4, i, 0);
    }

    public static void addVersion(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addInt(0, i, 1);
    }

    public static int createLodsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createTexturesVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int endModelDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static void finishModelDefBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finish(i);
    }

    public static void finishSizePrefixedModelDefBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finishSizePrefixed(i);
    }

    public static ModelDef getRootAsModelDef(ByteBuffer byteBuffer) {
        return getRootAsModelDef(byteBuffer, new ModelDef());
    }

    public static void startLodsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startModelDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(5);
    }

    public static void startTexturesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public ModelDef __assign(int i, ByteBuffer byteBuffer) {
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

    public AabbDef boundingBox() {
        return boundingBox(new AabbDef());
    }

    public ModelInstanceDef lods(int i) {
        return lods(new ModelInstanceDef(), i);
    }

    public int lodsLength() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public SkeletonDef skeleton() {
        return skeleton(new SkeletonDef());
    }

    public TextureDef textures(int i) {
        return textures(new TextureDef(), i);
    }

    public int texturesLength() {
        int __offset = __offset(12);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public int version() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.getInt(__offset + this.bb_pos);
        }
        return 1;
    }

    public static ModelDef getRootAsModelDef(ByteBuffer byteBuffer, ModelDef modelDef) {
        return modelDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public AabbDef boundingBox(AabbDef aabbDef) {
        int __offset = __offset(10);
        if (__offset != 0) {
            return aabbDef.__assign(__offset + this.bb_pos, this.bb);
        }
        return null;
    }

    public ModelInstanceDef lods(ModelInstanceDef modelInstanceDef, int i) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return modelInstanceDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public SkeletonDef skeleton(SkeletonDef skeletonDef) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return skeletonDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }

    public TextureDef textures(TextureDef textureDef, int i) {
        int __offset = __offset(12);
        if (__offset != 0) {
            return textureDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }
}