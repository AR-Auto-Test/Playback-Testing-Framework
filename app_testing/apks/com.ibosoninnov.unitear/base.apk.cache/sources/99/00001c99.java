package com.google.ar.sceneform.lullmodel;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ModelPipelineDef extends Table {
    public static void addCollidable(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static void addRenderables(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addSkeleton(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(3, i, 0);
    }

    public static void addSources(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addTextures(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(4, i, 0);
    }

    public static int createModelPipelineDef(FlatBufferBuilder flatBufferBuilder, int i, int i2, int i3, int i4, int i5) {
        flatBufferBuilder.startObject(5);
        addTextures(flatBufferBuilder, i5);
        addSkeleton(flatBufferBuilder, i4);
        addCollidable(flatBufferBuilder, i3);
        addRenderables(flatBufferBuilder, i2);
        addSources(flatBufferBuilder, i);
        return endModelPipelineDef(flatBufferBuilder);
    }

    public static int createRenderablesVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createSourcesVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
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

    public static int endModelPipelineDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ModelPipelineDef getRootAsModelPipelineDef(ByteBuffer byteBuffer) {
        return getRootAsModelPipelineDef(byteBuffer, new ModelPipelineDef());
    }

    public static void startModelPipelineDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(5);
    }

    public static void startRenderablesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startSourcesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startTexturesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public ModelPipelineDef __assign(int i, ByteBuffer byteBuffer) {
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

    public ModelPipelineCollidableDef collidable() {
        return collidable(new ModelPipelineCollidableDef());
    }

    public ModelPipelineRenderableDef renderables(int i) {
        return renderables(new ModelPipelineRenderableDef(), i);
    }

    public int renderablesLength() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public ModelPipelineSkeletonDef skeleton() {
        return skeleton(new ModelPipelineSkeletonDef());
    }

    public ModelPipelineImportDef sources(int i) {
        return sources(new ModelPipelineImportDef(), i);
    }

    public int sourcesLength() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
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

    public static ModelPipelineDef getRootAsModelPipelineDef(ByteBuffer byteBuffer, ModelPipelineDef modelPipelineDef) {
        return modelPipelineDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public ModelPipelineCollidableDef collidable(ModelPipelineCollidableDef modelPipelineCollidableDef) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return modelPipelineCollidableDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }

    public ModelPipelineRenderableDef renderables(ModelPipelineRenderableDef modelPipelineRenderableDef, int i) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return modelPipelineRenderableDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public ModelPipelineSkeletonDef skeleton(ModelPipelineSkeletonDef modelPipelineSkeletonDef) {
        int __offset = __offset(10);
        if (__offset != 0) {
            return modelPipelineSkeletonDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }

    public ModelPipelineImportDef sources(ModelPipelineImportDef modelPipelineImportDef, int i) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return modelPipelineImportDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
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