package com.google.ar.sceneform.rendering;

import com.google.android.filament.MaterialInstance;
import com.google.android.filament.TextureSampler;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.core.annotations.UsedByNative;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Texture;
import java.util.HashMap;

@UsedByNative("material_java_wrappers.h")
/* loaded from: classes.dex */
public final class MaterialParameters {
    private final HashMap<String, Parameter> namedParameters = new HashMap<>();

    /* renamed from: com.google.ar.sceneform.rendering.MaterialParameters$1  reason: invalid class name */
    /* loaded from: classes.dex */
    public static /* synthetic */ class AnonymousClass1 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$MagFilter;
        public static final /* synthetic */ int[] $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$MinFilter;
        public static final /* synthetic */ int[] $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$WrapMode;

        static {
            Texture.Sampler.WrapMode.values();
            int[] iArr = new int[3];
            $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$WrapMode = iArr;
            try {
                iArr[Texture.Sampler.WrapMode.CLAMP_TO_EDGE.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$WrapMode[Texture.Sampler.WrapMode.REPEAT.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$WrapMode[Texture.Sampler.WrapMode.MIRRORED_REPEAT.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
            Texture.Sampler.MagFilter.values();
            int[] iArr2 = new int[2];
            $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$MagFilter = iArr2;
            try {
                iArr2[Texture.Sampler.MagFilter.NEAREST.ordinal()] = 1;
            } catch (NoSuchFieldError unused4) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$MagFilter[Texture.Sampler.MagFilter.LINEAR.ordinal()] = 2;
            } catch (NoSuchFieldError unused5) {
            }
            Texture.Sampler.MinFilter.values();
            int[] iArr3 = new int[6];
            $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$MinFilter = iArr3;
            try {
                iArr3[Texture.Sampler.MinFilter.NEAREST.ordinal()] = 1;
            } catch (NoSuchFieldError unused6) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$MinFilter[Texture.Sampler.MinFilter.LINEAR.ordinal()] = 2;
            } catch (NoSuchFieldError unused7) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$MinFilter[Texture.Sampler.MinFilter.NEAREST_MIPMAP_NEAREST.ordinal()] = 3;
            } catch (NoSuchFieldError unused8) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$MinFilter[Texture.Sampler.MinFilter.LINEAR_MIPMAP_NEAREST.ordinal()] = 4;
            } catch (NoSuchFieldError unused9) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$MinFilter[Texture.Sampler.MinFilter.NEAREST_MIPMAP_LINEAR.ordinal()] = 5;
            } catch (NoSuchFieldError unused10) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$Texture$Sampler$MinFilter[Texture.Sampler.MinFilter.LINEAR_MIPMAP_LINEAR.ordinal()] = 6;
            } catch (NoSuchFieldError unused11) {
            }
        }
    }

    /* loaded from: classes.dex */
    public static class Boolean2Parameter extends Parameter {
        public boolean x;
        public boolean y;

        public Boolean2Parameter(String str, boolean z, boolean z2) {
            this.name = str;
            this.x = z;
            this.y = z2;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x, this.y);
        }
    }

    /* loaded from: classes.dex */
    public static class Boolean3Parameter extends Parameter {
        public boolean x;
        public boolean y;
        public boolean z;

        public Boolean3Parameter(String str, boolean z, boolean z2, boolean z3) {
            this.name = str;
            this.x = z;
            this.y = z2;
            this.z = z3;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x, this.y, this.z);
        }
    }

    /* loaded from: classes.dex */
    public static class Boolean4Parameter extends Parameter {
        public boolean w;
        public boolean x;
        public boolean y;
        public boolean z;

        public Boolean4Parameter(String str, boolean z, boolean z2, boolean z3, boolean z4) {
            this.name = str;
            this.x = z;
            this.y = z2;
            this.z = z3;
            this.w = z4;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x, this.y, this.z, this.w);
        }
    }

    /* loaded from: classes.dex */
    public static class BooleanParameter extends Parameter {
        public boolean x;

        public BooleanParameter(String str, boolean z) {
            this.name = str;
            this.x = z;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x);
        }
    }

    /* loaded from: classes.dex */
    public static class ExternalTextureParameter extends Parameter {
        private final ExternalTexture externalTexture;

        public ExternalTextureParameter(String str, ExternalTexture externalTexture) {
            this.name = str;
            this.externalTexture = externalTexture;
        }

        private TextureSampler getExternalFilamentSampler() {
            TextureSampler textureSampler = new TextureSampler();
            textureSampler.setMinFilter(TextureSampler.MinFilter.LINEAR);
            textureSampler.setMagFilter(TextureSampler.MagFilter.LINEAR);
            TextureSampler.WrapMode wrapMode = TextureSampler.WrapMode.CLAMP_TO_EDGE;
            textureSampler.setWrapModeS(wrapMode);
            textureSampler.setWrapModeT(wrapMode);
            textureSampler.setWrapModeR(wrapMode);
            return textureSampler;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.externalTexture.getFilamentTexture(), getExternalFilamentSampler());
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        /* renamed from: clone */
        public Parameter mo25clone() {
            return new ExternalTextureParameter(this.name, this.externalTexture);
        }
    }

    /* loaded from: classes.dex */
    public static class Float2Parameter extends Parameter {
        public float x;
        public float y;

        public Float2Parameter(String str, float f2, float f3) {
            this.name = str;
            this.x = f2;
            this.y = f3;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x, this.y);
        }
    }

    /* loaded from: classes.dex */
    public static class Float3Parameter extends Parameter {
        public float x;
        public float y;
        public float z;

        public Float3Parameter(String str, float f2, float f3, float f4) {
            this.name = str;
            this.x = f2;
            this.y = f3;
            this.z = f4;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x, this.y, this.z);
        }
    }

    /* loaded from: classes.dex */
    public static class Float4Parameter extends Parameter {
        public float w;
        public float x;
        public float y;
        public float z;

        public Float4Parameter(String str, float f2, float f3, float f4, float f5) {
            this.name = str;
            this.x = f2;
            this.y = f3;
            this.z = f4;
            this.w = f5;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x, this.y, this.z, this.w);
        }
    }

    /* loaded from: classes.dex */
    public static class FloatParameter extends Parameter {
        public float x;

        public FloatParameter(String str, float f2) {
            this.name = str;
            this.x = f2;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x);
        }
    }

    /* loaded from: classes.dex */
    public static class Int2Parameter extends Parameter {
        public int x;
        public int y;

        public Int2Parameter(String str, int i, int i2) {
            this.name = str;
            this.x = i;
            this.y = i2;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x, this.y);
        }
    }

    /* loaded from: classes.dex */
    public static class Int3Parameter extends Parameter {
        public int x;
        public int y;
        public int z;

        public Int3Parameter(String str, int i, int i2, int i3) {
            this.name = str;
            this.x = i;
            this.y = i2;
            this.z = i3;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x, this.y, this.z);
        }
    }

    /* loaded from: classes.dex */
    public static class Int4Parameter extends Parameter {
        public int w;
        public int x;
        public int y;
        public int z;

        public Int4Parameter(String str, int i, int i2, int i3, int i4) {
            this.name = str;
            this.x = i;
            this.y = i2;
            this.z = i3;
            this.w = i4;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x, this.y, this.z, this.w);
        }
    }

    /* loaded from: classes.dex */
    public static class IntParameter extends Parameter {
        public int x;

        public IntParameter(String str, int i) {
            this.name = str;
            this.x = i;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.x);
        }
    }

    /* loaded from: classes.dex */
    public static abstract class Parameter implements Cloneable {
        public String name;

        public abstract void applyTo(MaterialInstance materialInstance);

        /* JADX DEBUG: Method merged with bridge method */
        @Override // 
        /* renamed from: clone */
        public Parameter mo25clone() {
            try {
                return (Parameter) super.clone();
            } catch (CloneNotSupportedException e2) {
                throw new AssertionError(e2);
            }
        }
    }

    /* loaded from: classes.dex */
    public static class TextureParameter extends Parameter {
        public final Texture texture;

        public TextureParameter(String str, Texture texture) {
            this.name = str;
            this.texture = texture;
        }

        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        public void applyTo(MaterialInstance materialInstance) {
            materialInstance.setParameter(this.name, this.texture.getFilamentTexture(), MaterialParameters.convertTextureSampler(this.texture.getSampler()));
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // com.google.ar.sceneform.rendering.MaterialParameters.Parameter
        /* renamed from: clone */
        public Parameter mo25clone() {
            return new TextureParameter(this.name, this.texture);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static TextureSampler convertTextureSampler(Texture.Sampler sampler) {
        TextureSampler textureSampler = new TextureSampler();
        int ordinal = sampler.getMinFilter().ordinal();
        if (ordinal == 0) {
            textureSampler.setMinFilter(TextureSampler.MinFilter.NEAREST);
        } else if (ordinal == 1) {
            textureSampler.setMinFilter(TextureSampler.MinFilter.LINEAR);
        } else if (ordinal == 2) {
            textureSampler.setMinFilter(TextureSampler.MinFilter.NEAREST_MIPMAP_NEAREST);
        } else if (ordinal == 3) {
            textureSampler.setMinFilter(TextureSampler.MinFilter.LINEAR_MIPMAP_NEAREST);
        } else if (ordinal == 4) {
            textureSampler.setMinFilter(TextureSampler.MinFilter.NEAREST_MIPMAP_LINEAR);
        } else if (ordinal == 5) {
            textureSampler.setMinFilter(TextureSampler.MinFilter.LINEAR_MIPMAP_LINEAR);
        } else {
            throw new IllegalArgumentException("Invalid MinFilter");
        }
        int ordinal2 = sampler.getMagFilter().ordinal();
        if (ordinal2 == 0) {
            textureSampler.setMagFilter(TextureSampler.MagFilter.NEAREST);
        } else if (ordinal2 == 1) {
            textureSampler.setMagFilter(TextureSampler.MagFilter.LINEAR);
        } else {
            throw new IllegalArgumentException("Invalid MagFilter");
        }
        textureSampler.setWrapModeS(convertWrapMode(sampler.getWrapModeS()));
        textureSampler.setWrapModeT(convertWrapMode(sampler.getWrapModeT()));
        textureSampler.setWrapModeR(convertWrapMode(sampler.getWrapModeR()));
        return textureSampler;
    }

    private static TextureSampler.WrapMode convertWrapMode(Texture.Sampler.WrapMode wrapMode) {
        int ordinal = wrapMode.ordinal();
        if (ordinal != 0) {
            if (ordinal != 1) {
                if (ordinal == 2) {
                    return TextureSampler.WrapMode.MIRRORED_REPEAT;
                }
                throw new IllegalArgumentException("Invalid WrapMode");
            }
            return TextureSampler.WrapMode.REPEAT;
        }
        return TextureSampler.WrapMode.CLAMP_TO_EDGE;
    }

    public void applyTo(MaterialInstance materialInstance) {
        com.google.android.filament.Material material = materialInstance.getMaterial();
        for (Parameter parameter : this.namedParameters.values()) {
            if (material.hasParameter(parameter.name)) {
                parameter.applyTo(materialInstance);
            }
        }
    }

    public void copyFrom(MaterialParameters materialParameters) {
        this.namedParameters.clear();
        merge(materialParameters);
    }

    public boolean getBoolean(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof BooleanParameter) {
            return ((BooleanParameter) parameter).x;
        }
        return false;
    }

    public boolean[] getBoolean2(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof Boolean2Parameter) {
            Boolean2Parameter boolean2Parameter = (Boolean2Parameter) parameter;
            return new boolean[]{boolean2Parameter.x, boolean2Parameter.y};
        }
        return null;
    }

    public boolean[] getBoolean3(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof Boolean3Parameter) {
            Boolean3Parameter boolean3Parameter = (Boolean3Parameter) parameter;
            return new boolean[]{boolean3Parameter.x, boolean3Parameter.y, boolean3Parameter.z};
        }
        return null;
    }

    public boolean[] getBoolean4(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof Boolean4Parameter) {
            Boolean4Parameter boolean4Parameter = (Boolean4Parameter) parameter;
            return new boolean[]{boolean4Parameter.x, boolean4Parameter.y, boolean4Parameter.z, boolean4Parameter.w};
        }
        return null;
    }

    public ExternalTexture getExternalTexture(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof ExternalTextureParameter) {
            return ((ExternalTextureParameter) parameter).externalTexture;
        }
        return null;
    }

    public float getFloat(String str) {
        Parameter parameter = this.namedParameters.get(str);
        return parameter instanceof FloatParameter ? ((FloatParameter) parameter).x : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public float[] getFloat2(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof Float2Parameter) {
            Float2Parameter float2Parameter = (Float2Parameter) parameter;
            return new float[]{float2Parameter.x, float2Parameter.y};
        }
        return null;
    }

    public float[] getFloat3(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof Float3Parameter) {
            Float3Parameter float3Parameter = (Float3Parameter) parameter;
            return new float[]{float3Parameter.x, float3Parameter.y, float3Parameter.z};
        }
        return null;
    }

    public float[] getFloat4(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof Float4Parameter) {
            Float4Parameter float4Parameter = (Float4Parameter) parameter;
            return new float[]{float4Parameter.x, float4Parameter.y, float4Parameter.z, float4Parameter.w};
        }
        return null;
    }

    public int getInt(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof IntParameter) {
            return ((IntParameter) parameter).x;
        }
        return 0;
    }

    public int[] getInt2(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof Int2Parameter) {
            Int2Parameter int2Parameter = (Int2Parameter) parameter;
            return new int[]{int2Parameter.x, int2Parameter.y};
        }
        return null;
    }

    public int[] getInt3(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof Int3Parameter) {
            Int3Parameter int3Parameter = (Int3Parameter) parameter;
            return new int[]{int3Parameter.x, int3Parameter.y, int3Parameter.z};
        }
        return null;
    }

    public int[] getInt4(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof Int4Parameter) {
            Int4Parameter int4Parameter = (Int4Parameter) parameter;
            return new int[]{int4Parameter.x, int4Parameter.y, int4Parameter.z, int4Parameter.w};
        }
        return null;
    }

    public Texture getTexture(String str) {
        Parameter parameter = this.namedParameters.get(str);
        if (parameter instanceof TextureParameter) {
            return ((TextureParameter) parameter).texture;
        }
        return null;
    }

    public void merge(MaterialParameters materialParameters) {
        for (Parameter parameter : materialParameters.namedParameters.values()) {
            Parameter mo25clone = parameter.mo25clone();
            this.namedParameters.put(mo25clone.name, mo25clone);
        }
    }

    public void mergeIfAbsent(MaterialParameters materialParameters) {
        for (Parameter parameter : materialParameters.namedParameters.values()) {
            if (!this.namedParameters.containsKey(parameter.name)) {
                Parameter mo25clone = parameter.mo25clone();
                this.namedParameters.put(mo25clone.name, mo25clone);
            }
        }
    }

    @UsedByNative("material_java_wrappers.h")
    public void setBoolean(String str, boolean z) {
        this.namedParameters.put(str, new BooleanParameter(str, z));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setBoolean2(String str, boolean z, boolean z2) {
        this.namedParameters.put(str, new Boolean2Parameter(str, z, z2));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setBoolean3(String str, boolean z, boolean z2, boolean z3) {
        this.namedParameters.put(str, new Boolean3Parameter(str, z, z2, z3));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setBoolean4(String str, boolean z, boolean z2, boolean z3, boolean z4) {
        this.namedParameters.put(str, new Boolean4Parameter(str, z, z2, z3, z4));
    }

    public void setExternalTexture(String str, ExternalTexture externalTexture) {
        this.namedParameters.put(str, new ExternalTextureParameter(str, externalTexture));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setFloat(String str, float f2) {
        this.namedParameters.put(str, new FloatParameter(str, f2));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setFloat2(String str, float f2, float f3) {
        this.namedParameters.put(str, new Float2Parameter(str, f2, f3));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setFloat3(String str, float f2, float f3, float f4) {
        this.namedParameters.put(str, new Float3Parameter(str, f2, f3, f4));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setFloat4(String str, float f2, float f3, float f4, float f5) {
        this.namedParameters.put(str, new Float4Parameter(str, f2, f3, f4, f5));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setInt(String str, int i) {
        this.namedParameters.put(str, new IntParameter(str, i));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setInt2(String str, int i, int i2) {
        this.namedParameters.put(str, new Int2Parameter(str, i, i2));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setInt3(String str, int i, int i2, int i3) {
        this.namedParameters.put(str, new Int3Parameter(str, i, i2, i3));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setInt4(String str, int i, int i2, int i3, int i4) {
        this.namedParameters.put(str, new Int4Parameter(str, i, i2, i3, i4));
    }

    @UsedByNative("material_java_wrappers.h")
    public void setTexture(String str, Texture texture) {
        this.namedParameters.put(str, new TextureParameter(str, texture));
    }

    public void setFloat3(String str, Vector3 vector3) {
        this.namedParameters.put(str, new Float3Parameter(str, vector3.x, vector3.y, vector3.z));
    }
}