package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.ar.schemas.lull.ModelDef;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class SceneformBundleDef extends Table {
    public static boolean SceneformBundleDefBufferHasIdentifier(ByteBuffer byteBuffer) {
        return Table.__has_identifier(byteBuffer, "RBUN");
    }

    public static void addAnimations(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(8, i, 0);
    }

    public static void addCompiledMaterials(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(3, i, 0);
    }

    public static void addInputs(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(6, i, 0);
    }

    public static void addLightingDefs(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(7, i, 0);
    }

    public static void addMaterials(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static void addModel(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addRuntime(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(10, i, 0);
    }

    public static void addSamplers(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(5, i, 0);
    }

    public static void addSuggestedCollisionShape(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(4, i, 0);
    }

    public static void addTransform(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(9, i, 0);
    }

    public static void addVersion(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createAnimationsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createCompiledMaterialsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createInputsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createLightingDefsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createMaterialsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createSamplersVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createSceneformBundleDef(FlatBufferBuilder flatBufferBuilder, int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11) {
        flatBufferBuilder.startObject(11);
        addRuntime(flatBufferBuilder, i11);
        addTransform(flatBufferBuilder, i10);
        addAnimations(flatBufferBuilder, i9);
        addLightingDefs(flatBufferBuilder, i8);
        addInputs(flatBufferBuilder, i7);
        addSamplers(flatBufferBuilder, i6);
        addSuggestedCollisionShape(flatBufferBuilder, i5);
        addCompiledMaterials(flatBufferBuilder, i4);
        addMaterials(flatBufferBuilder, i3);
        addModel(flatBufferBuilder, i2);
        addVersion(flatBufferBuilder, i);
        return endSceneformBundleDef(flatBufferBuilder);
    }

    public static int endSceneformBundleDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static void finishSceneformBundleDefBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finish(i, "RBUN");
    }

    public static void finishSizePrefixedSceneformBundleDefBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finishSizePrefixed(i, "RBUN");
    }

    public static SceneformBundleDef getRootAsSceneformBundleDef(ByteBuffer byteBuffer) {
        return getRootAsSceneformBundleDef(byteBuffer, new SceneformBundleDef());
    }

    public static void startAnimationsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startCompiledMaterialsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startInputsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startLightingDefsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startMaterialsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startSamplersVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startSceneformBundleDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(11);
    }

    public SceneformBundleDef __assign(int i, ByteBuffer byteBuffer) {
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

    public AnimationDef animations(int i) {
        return animations(new AnimationDef(), i);
    }

    public int animationsLength() {
        int __offset = __offset(20);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public CompiledMaterialDef compiledMaterials(int i) {
        return compiledMaterials(new CompiledMaterialDef(), i);
    }

    public int compiledMaterialsLength() {
        int __offset = __offset(10);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public InputDef inputs(int i) {
        return inputs(new InputDef(), i);
    }

    public int inputsLength() {
        int __offset = __offset(16);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public LightingDef lightingDefs(int i) {
        return lightingDefs(new LightingDef(), i);
    }

    public int lightingDefsLength() {
        int __offset = __offset(18);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public MaterialDef materials(int i) {
        return materials(new MaterialDef(), i);
    }

    public int materialsLength() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public ModelDef model() {
        return model(new ModelDef());
    }

    public RuntimeAssetDef runtime() {
        return runtime(new RuntimeAssetDef());
    }

    public SamplerDef samplers(int i) {
        return samplers(new SamplerDef(), i);
    }

    public int samplersLength() {
        int __offset = __offset(14);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public SuggestedCollisionShapeDef suggestedCollisionShape() {
        return suggestedCollisionShape(new SuggestedCollisionShapeDef());
    }

    public TransformDef transform() {
        return transform(new TransformDef());
    }

    public VersionDef version() {
        return version(new VersionDef());
    }

    public static SceneformBundleDef getRootAsSceneformBundleDef(ByteBuffer byteBuffer, SceneformBundleDef sceneformBundleDef) {
        return sceneformBundleDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public AnimationDef animations(AnimationDef animationDef, int i) {
        int __offset = __offset(20);
        if (__offset != 0) {
            return animationDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public CompiledMaterialDef compiledMaterials(CompiledMaterialDef compiledMaterialDef, int i) {
        int __offset = __offset(10);
        if (__offset != 0) {
            return compiledMaterialDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public InputDef inputs(InputDef inputDef, int i) {
        int __offset = __offset(16);
        if (__offset != 0) {
            return inputDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public LightingDef lightingDefs(LightingDef lightingDef, int i) {
        int __offset = __offset(18);
        if (__offset != 0) {
            return lightingDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public MaterialDef materials(MaterialDef materialDef, int i) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return materialDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public ModelDef model(ModelDef modelDef) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return modelDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }

    public RuntimeAssetDef runtime(RuntimeAssetDef runtimeAssetDef) {
        int __offset = __offset(24);
        if (__offset != 0) {
            return runtimeAssetDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }

    public SamplerDef samplers(SamplerDef samplerDef, int i) {
        int __offset = __offset(14);
        if (__offset != 0) {
            return samplerDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public SuggestedCollisionShapeDef suggestedCollisionShape(SuggestedCollisionShapeDef suggestedCollisionShapeDef) {
        int __offset = __offset(12);
        if (__offset != 0) {
            return suggestedCollisionShapeDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }

    public TransformDef transform(TransformDef transformDef) {
        int __offset = __offset(22);
        if (__offset != 0) {
            return transformDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }

    public VersionDef version(VersionDef versionDef) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return versionDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }
}