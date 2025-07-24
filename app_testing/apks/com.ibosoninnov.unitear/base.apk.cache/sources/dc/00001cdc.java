package com.google.ar.sceneform.rendering;

import android.net.Uri;
import android.util.Log;
import c.b.a.a.a;
import c.d.b.a.q.p;
import c.d.b.a.q.r;
import com.google.android.filament.IndexBuffer;
import com.google.android.filament.TextureSampler;
import com.google.android.filament.VertexBuffer;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.LoadRenderableFromSfbTask;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.RenderableInternalData;
import com.google.ar.sceneform.rendering.SceneformBundle;
import com.google.ar.sceneform.rendering.Texture;
import com.google.ar.sceneform.utilities.Preconditions;
import com.google.ar.sceneform.utilities.SceneformBufferUtils;
import com.google.ar.schemas.lull.ModelDef;
import com.google.ar.schemas.lull.ModelIndexRange;
import com.google.ar.schemas.lull.ModelInstanceDef;
import com.google.ar.schemas.lull.Vec3;
import com.google.ar.schemas.lull.VertexAttribute;
import com.google.ar.schemas.sceneform.BoolInit;
import com.google.ar.schemas.sceneform.BoolVec2Init;
import com.google.ar.schemas.sceneform.BoolVec3Init;
import com.google.ar.schemas.sceneform.BoolVec4Init;
import com.google.ar.schemas.sceneform.CompiledMaterialDef;
import com.google.ar.schemas.sceneform.IntInit;
import com.google.ar.schemas.sceneform.IntVec2Init;
import com.google.ar.schemas.sceneform.IntVec3Init;
import com.google.ar.schemas.sceneform.IntVec4Init;
import com.google.ar.schemas.sceneform.MaterialDef;
import com.google.ar.schemas.sceneform.ParameterDef;
import com.google.ar.schemas.sceneform.ParameterInitDef;
import com.google.ar.schemas.sceneform.SamplerDef;
import com.google.ar.schemas.sceneform.SamplerInit;
import com.google.ar.schemas.sceneform.ScalarInit;
import com.google.ar.schemas.sceneform.SceneformBundleDef;
import com.google.ar.schemas.sceneform.TransformDef;
import com.google.ar.schemas.sceneform.Vec2Init;
import com.google.ar.schemas.sceneform.Vec3Init;
import com.google.ar.schemas.sceneform.Vec4Init;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CompletionStage;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

/* loaded from: classes.dex */
public class LoadRenderableFromSfbTask<T extends Renderable> {
    private static final int BYTES_PER_FLOAT = 4;
    private static final int BYTES_PER_INT = 4;
    private static final int BYTES_PER_SHORT = 2;
    private static final String TAG = "LoadRenderableFromSfbTask";
    private ByteBuffer indexBufferData;
    private int indexCount;
    private IndexBuffer.Builder.IndexType indexType;
    private int meshCount;
    private ModelDef modelDef;
    private ModelInstanceDef modelInstanceDef;
    private final T renderable;
    private final RenderableInternalData renderableData;
    private final Uri renderableUri;
    private int textureCount;
    private TransformDef transformDef;
    private ByteBuffer vertexBufferData;
    private int vertexCount;
    private int vertexStride;
    private final ArrayList<ModelTexture> textures = new ArrayList<>();
    private final ArrayList<Material> compiledMaterials = new ArrayList<>();
    private final ArrayList<Integer> compiledMaterialIndex = new ArrayList<>();
    private final ArrayList<MaterialParameters> materialParameters = new ArrayList<>();
    private final ArrayList<String> materialNames = new ArrayList<>();

    /* renamed from: com.google.ar.sceneform.rendering.LoadRenderableFromSfbTask$1  reason: invalid class name */
    /* loaded from: classes.dex */
    public static /* synthetic */ class AnonymousClass1 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$android$filament$TextureSampler$MagFilter;
        public static final /* synthetic */ int[] $SwitchMap$com$google$android$filament$TextureSampler$MinFilter;
        public static final /* synthetic */ int[] $SwitchMap$com$google$android$filament$TextureSampler$WrapMode;

        static {
            TextureSampler.WrapMode.values();
            int[] iArr = new int[3];
            $SwitchMap$com$google$android$filament$TextureSampler$WrapMode = iArr;
            try {
                iArr[TextureSampler.WrapMode.CLAMP_TO_EDGE.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$android$filament$TextureSampler$WrapMode[TextureSampler.WrapMode.REPEAT.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                $SwitchMap$com$google$android$filament$TextureSampler$WrapMode[TextureSampler.WrapMode.MIRRORED_REPEAT.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
            TextureSampler.MinFilter.values();
            int[] iArr2 = new int[6];
            $SwitchMap$com$google$android$filament$TextureSampler$MinFilter = iArr2;
            try {
                iArr2[TextureSampler.MinFilter.NEAREST.ordinal()] = 1;
            } catch (NoSuchFieldError unused4) {
            }
            try {
                $SwitchMap$com$google$android$filament$TextureSampler$MinFilter[TextureSampler.MinFilter.LINEAR.ordinal()] = 2;
            } catch (NoSuchFieldError unused5) {
            }
            try {
                $SwitchMap$com$google$android$filament$TextureSampler$MinFilter[TextureSampler.MinFilter.NEAREST_MIPMAP_NEAREST.ordinal()] = 3;
            } catch (NoSuchFieldError unused6) {
            }
            try {
                $SwitchMap$com$google$android$filament$TextureSampler$MinFilter[TextureSampler.MinFilter.LINEAR_MIPMAP_NEAREST.ordinal()] = 4;
            } catch (NoSuchFieldError unused7) {
            }
            try {
                $SwitchMap$com$google$android$filament$TextureSampler$MinFilter[TextureSampler.MinFilter.NEAREST_MIPMAP_LINEAR.ordinal()] = 5;
            } catch (NoSuchFieldError unused8) {
            }
            try {
                $SwitchMap$com$google$android$filament$TextureSampler$MinFilter[TextureSampler.MinFilter.LINEAR_MIPMAP_LINEAR.ordinal()] = 6;
            } catch (NoSuchFieldError unused9) {
            }
            TextureSampler.MagFilter.values();
            int[] iArr3 = new int[2];
            $SwitchMap$com$google$android$filament$TextureSampler$MagFilter = iArr3;
            try {
                iArr3[TextureSampler.MagFilter.NEAREST.ordinal()] = 1;
            } catch (NoSuchFieldError unused10) {
            }
            try {
                $SwitchMap$com$google$android$filament$TextureSampler$MagFilter[TextureSampler.MagFilter.LINEAR.ordinal()] = 2;
            } catch (NoSuchFieldError unused11) {
            }
        }
    }

    /* loaded from: classes.dex */
    public static class ModelTexture {
        public Texture data = null;
        public String name;

        public ModelTexture(String str) {
            this.name = str;
        }
    }

    public LoadRenderableFromSfbTask(T t, Uri uri) {
        this.renderable = t;
        IRenderableInternalData renderableData = t.getRenderableData();
        if (renderableData instanceof RenderableInternalData) {
            this.renderableData = (RenderableInternalData) renderableData;
            this.renderableUri = uri;
            return;
        }
        StringBuilder x = a.x("Expected task type ");
        x.append(TAG);
        throw new IllegalStateException(x.toString());
    }

    private void buildGeometry() {
        ByteBuffer vertexDataAsByteBuffer = this.modelInstanceDef.vertexDataAsByteBuffer();
        Preconditions.checkNotNull(vertexDataAsByteBuffer, "Model Instance geometry data is invalid (vertexData is null).");
        int vertexDataLength = this.modelInstanceDef.vertexDataLength();
        this.meshCount = this.modelInstanceDef.rangesLength();
        this.vertexCount = vertexDataLength / LullModel.getByteCountPerVertex(this.modelInstanceDef);
        if (this.modelInstanceDef.indices32Length() > 0) {
            int indices32Length = this.modelInstanceDef.indices32Length();
            this.indexCount = indices32Length;
            this.indexType = IndexBuffer.Builder.IndexType.UINT;
            ByteBuffer allocateDirect = ByteBuffer.allocateDirect(indices32Length * 4);
            this.indexBufferData = allocateDirect;
            allocateDirect.put(this.modelInstanceDef.indices32AsByteBuffer());
        } else if (this.modelInstanceDef.indices16Length() > 0) {
            int indices16Length = this.modelInstanceDef.indices16Length();
            this.indexCount = indices16Length;
            this.indexType = IndexBuffer.Builder.IndexType.USHORT;
            ByteBuffer allocateDirect2 = ByteBuffer.allocateDirect(indices16Length * 2);
            this.indexBufferData = allocateDirect2;
            allocateDirect2.put(this.modelInstanceDef.indices16AsByteBuffer());
        } else {
            throw new AssertionError("Model Instance geometry data is invalid (model has no index data).");
        }
        this.indexBufferData.flip();
        ByteBuffer allocateDirect3 = ByteBuffer.allocateDirect(vertexDataAsByteBuffer.remaining());
        this.vertexBufferData = allocateDirect3;
        Preconditions.checkNotNull(allocateDirect3, "Failed to allocate geometry for FilamentModel.");
        this.vertexBufferData.put(vertexDataAsByteBuffer);
        this.vertexBufferData.flip();
        this.vertexStride = 0;
        int vertexAttributesLength = this.modelInstanceDef.vertexAttributesLength();
        for (int i = 0; i < vertexAttributesLength; i++) {
            this.vertexStride += getVertexAttributeTypeSizeInBytes(this.modelInstanceDef.vertexAttributes(i).type());
        }
    }

    /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
    private SceneformBundleDef buildMaterialParameters(SceneformBundleDef sceneformBundleDef) {
        int i;
        int i2;
        LoadRenderableFromSfbTask<T> loadRenderableFromSfbTask;
        IntVec3Init intVec3Init;
        IntVec2Init intVec2Init;
        IntInit intInit;
        IntVec4Init intVec4Init;
        ParameterDef parameterDef;
        int i3;
        ScalarInit scalarInit;
        int i4;
        LoadRenderableFromSfbTask<T> loadRenderableFromSfbTask2 = this;
        SceneformBundleDef sceneformBundleDef2 = sceneformBundleDef;
        int materialsLength = sceneformBundleDef.materialsLength();
        if (materialsLength == 0) {
            Log.i(TAG, "Building materials but the sceneform bundle has no materials");
            return sceneformBundleDef2;
        }
        int i5 = 0;
        while (i5 < loadRenderableFromSfbTask2.meshCount) {
            MaterialDef materials = sceneformBundleDef2.materials(materialsLength <= i5 ? materialsLength - 1 : i5);
            if (materials == null) {
                String str = TAG;
                Log.e(str, "Material " + i5 + " is null.");
                loadRenderableFromSfbTask = loadRenderableFromSfbTask2;
                i = materialsLength;
                i2 = i5;
            } else {
                loadRenderableFromSfbTask2.compiledMaterialIndex.add(Integer.valueOf(materials.compiledIndex()));
                ParameterDef parameterDef2 = new ParameterDef();
                ParameterInitDef parameterInitDef = new ParameterInitDef();
                ScalarInit scalarInit2 = new ScalarInit();
                Vec2Init vec2Init = new Vec2Init();
                Vec3Init vec3Init = new Vec3Init();
                Vec4Init vec4Init = new Vec4Init();
                BoolInit boolInit = new BoolInit();
                BoolVec2Init boolVec2Init = new BoolVec2Init();
                BoolVec3Init boolVec3Init = new BoolVec3Init();
                BoolVec4Init boolVec4Init = new BoolVec4Init();
                IntInit intInit2 = new IntInit();
                i = materialsLength;
                IntVec2Init intVec2Init2 = new IntVec2Init();
                IntVec3Init intVec3Init2 = new IntVec3Init();
                i2 = i5;
                IntVec4Init intVec4Init2 = new IntVec4Init();
                ScalarInit scalarInit3 = scalarInit2;
                SamplerInit samplerInit = new SamplerInit();
                MaterialParameters materialParameters = new MaterialParameters();
                int parametersLength = materials.parametersLength();
                int i6 = 0;
                while (i6 < parametersLength) {
                    materials.parameters(parameterDef2, i6);
                    parameterDef2.initialValue(parameterInitDef);
                    int i7 = i6;
                    String id = parameterDef2.id();
                    switch (parameterInitDef.initType()) {
                        case 1:
                        case 16:
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            break;
                        case 2:
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            parameterInitDef.init(scalarInit);
                            materialParameters.setFloat(id, scalarInit.value());
                            break;
                        case 3:
                            intVec3Init = intVec3Init2;
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            i4 = i7;
                            parameterInitDef.init(vec3Init);
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            materialParameters.setFloat3(id, vec3Init.x(), vec3Init.y(), vec3Init.z());
                            scalarInit = scalarInit3;
                            break;
                        case 4:
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            i4 = i7;
                            parameterInitDef.init(vec4Init);
                            materialParameters.setFloat4(id, vec4Init.x(), vec4Init.y(), vec4Init.z(), vec4Init.w());
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            scalarInit = scalarInit3;
                            break;
                        case 5:
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            parameterInitDef.init(samplerInit);
                            i4 = i7;
                            Texture textureByName = getTextureByName(samplerInit.path());
                            if (textureByName != null) {
                                materialParameters.setTexture(id, textureByName);
                            }
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            scalarInit = scalarInit3;
                            break;
                        case 6:
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            parameterInitDef.init(vec2Init);
                            materialParameters.setFloat2(id, vec2Init.x(), vec2Init.y());
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            break;
                        case 7:
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            parameterInitDef.init(boolInit);
                            materialParameters.setBoolean(id, boolInit.value());
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            break;
                        case 8:
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            parameterInitDef.init(boolVec2Init);
                            materialParameters.setBoolean2(id, boolVec2Init.x(), boolVec2Init.y());
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            break;
                        case 9:
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            parameterInitDef.init(boolVec3Init);
                            materialParameters.setBoolean3(id, boolVec3Init.x(), boolVec3Init.y(), boolVec3Init.z());
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            break;
                        case 10:
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            parameterInitDef.init(boolVec4Init);
                            materialParameters.setBoolean4(id, boolVec4Init.x(), boolVec4Init.y(), boolVec4Init.z(), boolVec4Init.w());
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            break;
                        case 11:
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            parameterInitDef.init(intInit2);
                            materialParameters.setInt(id, intInit2.value());
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            break;
                        case 12:
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            parameterInitDef.init(intVec2Init2);
                            materialParameters.setInt2(id, intVec2Init2.x(), intVec2Init2.y());
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            break;
                        case 13:
                            parameterInitDef.init(intVec3Init2);
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            materialParameters.setInt3(id, intVec3Init2.x(), intVec3Init2.y(), intVec3Init2.z());
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            break;
                        case 14:
                            parameterInitDef.init(intVec4Init2);
                            materialParameters.setInt4(id, intVec4Init2.x(), intVec4Init2.y(), intVec4Init2.z(), intVec4Init2.w());
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            break;
                        case 15:
                        default:
                            intVec3Init = intVec3Init2;
                            intVec2Init = intVec2Init2;
                            intInit = intInit2;
                            intVec4Init = intVec4Init2;
                            parameterDef = parameterDef2;
                            i3 = parametersLength;
                            scalarInit = scalarInit3;
                            i4 = i7;
                            String str2 = TAG;
                            Log.e(str2, "Unknown parameter type: " + id);
                            break;
                    }
                    i6 = i4 + 1;
                    scalarInit3 = scalarInit;
                    intVec3Init2 = intVec3Init;
                    intVec2Init2 = intVec2Init;
                    intInit2 = intInit;
                    intVec4Init2 = intVec4Init;
                    parameterDef2 = parameterDef;
                    parametersLength = i3;
                }
                loadRenderableFromSfbTask = this;
                loadRenderableFromSfbTask.materialParameters.add(materialParameters);
                String name = materials.name();
                ArrayList<String> arrayList = loadRenderableFromSfbTask.materialNames;
                if (name == null) {
                    name = "";
                }
                arrayList.add(name);
            }
            i5 = i2 + 1;
            sceneformBundleDef2 = sceneformBundleDef;
            loadRenderableFromSfbTask2 = loadRenderableFromSfbTask;
            materialsLength = i;
        }
        return sceneformBundleDef;
    }

    private SceneformBundleDef byteBufferToSfb(ByteBuffer byteBuffer) {
        try {
            SceneformBundleDef tryLoadSceneformBundle = SceneformBundle.tryLoadSceneformBundle(byteBuffer);
            if (tryLoadSceneformBundle != null) {
                return tryLoadSceneformBundle;
            }
            StringBuilder x = a.x("No RCB file at uri: ");
            x.append(this.renderableUri);
            throw new AssertionError(x.toString());
        } catch (SceneformBundle.VersionException e2) {
            throw new CompletionException(e2);
        }
    }

    private static Texture.Sampler.WrapMode filamentWrapModeToWrapMode(TextureSampler.WrapMode wrapMode) {
        int ordinal = wrapMode.ordinal();
        if (ordinal != 0) {
            if (ordinal != 1) {
                if (ordinal == 2) {
                    return Texture.Sampler.WrapMode.MIRRORED_REPEAT;
                }
                throw new IllegalArgumentException("Invalid WrapMode");
            }
            return Texture.Sampler.WrapMode.REPEAT;
        }
        return Texture.Sampler.WrapMode.CLAMP_TO_EDGE;
    }

    private static VertexBuffer.AttributeType getFilamentAttributeType(int i) {
        switch (i) {
            case 1:
                return VertexBuffer.AttributeType.FLOAT;
            case 2:
                return VertexBuffer.AttributeType.FLOAT2;
            case 3:
                return VertexBuffer.AttributeType.FLOAT3;
            case 4:
                return VertexBuffer.AttributeType.FLOAT4;
            case 5:
                return VertexBuffer.AttributeType.USHORT2;
            case 6:
                return VertexBuffer.AttributeType.USHORT4;
            case 7:
                return VertexBuffer.AttributeType.UBYTE4;
            default:
                throw new AssertionError(a.j("Unsupported VertexAttributeType value: ", i));
        }
    }

    private static VertexBuffer.VertexAttribute getFilamentVertexAttribute(int i) {
        if (i != 1) {
            if (i != 2) {
                if (i != 3) {
                    if (i != 6) {
                        if (i != 7) {
                            if (i != 8) {
                                return null;
                            }
                            return VertexBuffer.VertexAttribute.BONE_WEIGHTS;
                        }
                        return VertexBuffer.VertexAttribute.BONE_INDICES;
                    }
                    return VertexBuffer.VertexAttribute.TANGENTS;
                }
                return VertexBuffer.VertexAttribute.UV0;
            }
            return VertexBuffer.VertexAttribute.COLOR;
        }
        return VertexBuffer.VertexAttribute.POSITION;
    }

    private Texture getTextureByName(String str) {
        for (int i = 0; i < this.textureCount; i++) {
            if (Objects.equals(str, this.textures.get(i).name)) {
                return this.textures.get(i).data;
            }
        }
        return null;
    }

    private static int getVertexAttributeTypeSizeInBytes(int i) {
        switch (i) {
            case 0:
                return 0;
            case 1:
            case 5:
            case 7:
                return 4;
            case 2:
            case 6:
                return 8;
            case 3:
                return 12;
            case 4:
                return 16;
            default:
                throw new AssertionError(a.j("Unsupported VertexAttributeType value: ", i));
        }
    }

    private boolean isAttributeNormalized(int i) {
        return i == 2 || i == 8;
    }

    private void loadAnimations(SceneformBundleDef sceneformBundleDef) {
    }

    private SceneformBundleDef loadModel(SceneformBundleDef sceneformBundleDef) {
        this.transformDef = sceneformBundleDef.transform();
        ModelDef model = sceneformBundleDef.model();
        this.modelDef = model;
        Preconditions.checkNotNull(model, "Model error: ModelDef is invalid.");
        ModelInstanceDef lods = this.modelDef.lods(0);
        this.modelInstanceDef = lods;
        Preconditions.checkNotNull(lods, "Lull Model error: ModelInstanceDef is invalid.");
        buildGeometry();
        return sceneformBundleDef;
    }

    /* JADX DEBUG: Type inference failed for r12v1. Raw type applied. Possible types: java.util.concurrent.CompletableFuture<U>, java.util.concurrent.CompletableFuture<com.google.ar.schemas.sceneform.SceneformBundleDef> */
    private CompletableFuture<SceneformBundleDef> loadTexturesAsync(final SceneformBundleDef sceneformBundleDef) {
        int samplersLength = sceneformBundleDef.samplersLength();
        this.textureCount = samplersLength;
        CompletableFuture[] completableFutureArr = new CompletableFuture[samplersLength];
        for (int i = 0; i < this.textureCount; i++) {
            SamplerDef samplers = sceneformBundleDef.samplers(i);
            final ModelTexture modelTexture = new ModelTexture(samplers.name());
            this.textures.add(modelTexture);
            int usageType = samplers.params().usageType();
            Texture.Usage[] values = Texture.Usage.values();
            if (usageType < 3) {
                Texture.Usage usage = values[usageType];
                if (samplers.dataLength() != 0) {
                    ByteBuffer dataAsByteBuffer = samplers.dataAsByteBuffer();
                    final ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(dataAsByteBuffer.array(), dataAsByteBuffer.arrayOffset(), dataAsByteBuffer.capacity());
                    boolean z = usage == Texture.Usage.COLOR;
                    byteArrayInputStream.skip(dataAsByteBuffer.position());
                    completableFutureArr[i] = Texture.builder().setUsage(usage).setSampler(samplerDefToSampler(samplers)).setPremultiplied(z).setSource(new Callable() { // from class: c.d.b.a.q.n
                        @Override // java.util.concurrent.Callable
                        public final Object call() {
                            ByteArrayInputStream byteArrayInputStream2 = byteArrayInputStream;
                            Preconditions.checkNotNull(byteArrayInputStream2);
                            return byteArrayInputStream2;
                        }
                    }).build().thenAccept(new Consumer() { // from class: c.d.b.a.q.m
                        @Override // java.util.function.Consumer
                        public final void accept(Object obj) {
                            LoadRenderableFromSfbTask.ModelTexture.this.data = (Texture) obj;
                        }
                    }).exceptionally((Function<Throwable, ? extends Void>) p.f4348a);
                } else {
                    throw new IllegalStateException("Unable to load texture, no sampler definition.");
                }
            } else {
                throw new AssertionError(a.j("Invalid Texture Usage: ", usageType));
            }
        }
        return CompletableFuture.allOf(completableFutureArr).thenApply(new Function() { // from class: c.d.b.a.q.o
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                Void r2 = (Void) obj;
                return SceneformBundleDef.this;
            }
        });
    }

    private static Texture.Sampler.MagFilter samplerDefToMagFilter(SamplerDef samplerDef) {
        int ordinal = TextureSampler.MagFilter.values()[samplerDef.params().magFilter()].ordinal();
        if (ordinal != 0) {
            if (ordinal == 1) {
                return Texture.Sampler.MagFilter.LINEAR;
            }
            throw new IllegalArgumentException("Invalid MagFilter");
        }
        return Texture.Sampler.MagFilter.NEAREST;
    }

    private static Texture.Sampler.MinFilter samplerDefToMinFilter(SamplerDef samplerDef) {
        int ordinal = TextureSampler.MinFilter.values()[samplerDef.params().minFilter()].ordinal();
        if (ordinal != 0) {
            if (ordinal != 1) {
                if (ordinal != 2) {
                    if (ordinal != 3) {
                        if (ordinal != 4) {
                            if (ordinal == 5) {
                                return Texture.Sampler.MinFilter.LINEAR_MIPMAP_LINEAR;
                            }
                            throw new IllegalArgumentException("Invalid MinFilter");
                        }
                        return Texture.Sampler.MinFilter.NEAREST_MIPMAP_LINEAR;
                    }
                    return Texture.Sampler.MinFilter.LINEAR_MIPMAP_NEAREST;
                }
                return Texture.Sampler.MinFilter.NEAREST_MIPMAP_NEAREST;
            }
            return Texture.Sampler.MinFilter.LINEAR;
        }
        return Texture.Sampler.MinFilter.NEAREST;
    }

    private static Texture.Sampler samplerDefToSampler(SamplerDef samplerDef) {
        Texture.Sampler.WrapMode filamentWrapModeToWrapMode = filamentWrapModeToWrapMode(TextureSampler.WrapMode.values()[samplerDef.params().wrapR()]);
        Texture.Sampler.WrapMode filamentWrapModeToWrapMode2 = filamentWrapModeToWrapMode(TextureSampler.WrapMode.values()[samplerDef.params().wrapS()]);
        return Texture.Sampler.builder().setMinFilter(samplerDefToMinFilter(samplerDef)).setMagFilter(samplerDefToMagFilter(samplerDef)).setWrapModeR(filamentWrapModeToWrapMode).setWrapModeS(filamentWrapModeToWrapMode2).setWrapModeT(filamentWrapModeToWrapMode(TextureSampler.WrapMode.values()[samplerDef.params().wrapT()])).build();
    }

    private SceneformBundleDef setCollisionShape(SceneformBundleDef sceneformBundleDef) {
        try {
            this.renderable.collisionShape = SceneformBundle.readCollisionGeometry(sceneformBundleDef);
            return sceneformBundleDef;
        } catch (IOException e2) {
            throw new CompletionException("Unable to get collision geometry from sfb", e2);
        }
    }

    private void setupAnimation() {
    }

    private T setupFilament(SceneformBundleDef sceneformBundleDef) {
        Preconditions.checkNotNull(sceneformBundleDef);
        setupFilamentGeometryBuffers();
        setupFilamentMaterials(sceneformBundleDef);
        setupRenderableData();
        this.renderable.getId().update();
        return this.renderable;
    }

    private void setupFilamentGeometryBuffers() {
        IEngine engine = EngineInstance.getEngine();
        IndexBuffer build = new IndexBuffer.Builder().indexCount(this.indexCount).bufferType(this.indexType).build(engine.getFilamentEngine());
        build.setBuffer(engine.getFilamentEngine(), this.indexBufferData);
        this.renderableData.setIndexBuffer(build);
        VertexBuffer.Builder bufferCount = new VertexBuffer.Builder().vertexCount(this.vertexCount).bufferCount(1);
        int vertexAttributesLength = this.modelInstanceDef.vertexAttributesLength();
        int i = 0;
        for (int i2 = 0; i2 < vertexAttributesLength; i2++) {
            VertexAttribute vertexAttributes = this.modelInstanceDef.vertexAttributes(i2);
            VertexBuffer.VertexAttribute filamentVertexAttribute = getFilamentVertexAttribute(vertexAttributes.usage());
            if (filamentVertexAttribute != null) {
                bufferCount.attribute(filamentVertexAttribute, 0, getFilamentAttributeType(vertexAttributes.type()), i, this.vertexStride);
                if (isAttributeNormalized(vertexAttributes.usage())) {
                    bufferCount.normalized(filamentVertexAttribute);
                }
            }
            i += getVertexAttributeTypeSizeInBytes(vertexAttributes.type());
        }
        VertexBuffer build2 = bufferCount.build(engine.getFilamentEngine());
        build2.setBufferAt(engine.getFilamentEngine(), 0, this.vertexBufferData);
        this.renderableData.setVertexBuffer(build2);
        setupAnimation();
    }

    private void setupFilamentMaterials(SceneformBundleDef sceneformBundleDef) {
        int compiledMaterialsLength = sceneformBundleDef.compiledMaterialsLength();
        for (int i = 0; i < compiledMaterialsLength; i++) {
            CompiledMaterialDef compiledMaterials = sceneformBundleDef.compiledMaterials(i);
            int hashCode = compiledMaterials.compiledMaterialAsByteBuffer().hashCode();
            try {
                Material now = Material.builder().setSource(SceneformBufferUtils.copyByteBuffer(compiledMaterials.compiledMaterialAsByteBuffer())).setRegistryId(Integer.valueOf(hashCode)).build().getNow(null);
                if (now != null) {
                    this.compiledMaterials.add(now);
                } else {
                    throw new AssertionError("Material wasn't loaded.");
                }
            } catch (IOException e2) {
                throw new CompletionException("Failed to create material", e2);
            }
        }
    }

    private void setupRenderableData() {
        Vec3 min = this.modelDef.boundingBox().min();
        Vector3 vector3 = new Vector3(min.x(), min.y(), min.z());
        Vec3 max = this.modelDef.boundingBox().max();
        Vector3 scaled = Vector3.subtract(new Vector3(max.x(), max.y(), max.z()), vector3).scaled(0.5f);
        Vector3 add = Vector3.add(vector3, scaled);
        this.renderableData.setExtentsAabb(scaled);
        this.renderableData.setCenterAabb(add);
        TransformDef transformDef = this.transformDef;
        if (transformDef != null && transformDef.scale() != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            Vec3 offset = this.transformDef.offset();
            Vector3 vector32 = new Vector3(offset.x(), offset.y(), offset.z());
            this.renderableData.setTransformScale(this.transformDef.scale());
            this.renderableData.setTransformOffset(vector32);
        }
        ArrayList<Material> materialBindings = this.renderable.getMaterialBindings();
        ArrayList<String> materialNames = this.renderable.getMaterialNames();
        materialBindings.clear();
        materialNames.clear();
        for (int i = 0; i < this.meshCount; i++) {
            ModelIndexRange ranges = this.modelInstanceDef.ranges(i);
            int end = (int) ranges.end();
            Material makeCopy = this.compiledMaterials.get(this.compiledMaterialIndex.get(i).intValue()).makeCopy();
            makeCopy.copyMaterialParameters(this.materialParameters.get(i));
            RenderableInternalData.MeshData meshData = new RenderableInternalData.MeshData();
            materialBindings.add(makeCopy);
            materialNames.add(this.materialNames.get(i));
            meshData.indexStart = (int) ranges.start();
            meshData.indexEnd = end;
            this.renderableData.getMeshes().add(meshData);
        }
    }

    public /* synthetic */ SceneformBundleDef a(Callable callable) {
        SceneformBundleDef byteBufferToSfb = byteBufferToSfb(SceneformBufferUtils.inputStreamToByteBuffer(callable));
        setCollisionShape(byteBufferToSfb);
        loadModel(byteBufferToSfb);
        return byteBufferToSfb;
    }

    public /* synthetic */ CompletionStage b(SceneformBundleDef sceneformBundleDef) {
        loadAnimations(sceneformBundleDef);
        return loadTexturesAsync(sceneformBundleDef);
    }

    public /* synthetic */ Renderable c(SceneformBundleDef sceneformBundleDef) {
        buildMaterialParameters(sceneformBundleDef);
        return setupFilament(sceneformBundleDef);
    }

    public CompletableFuture<T> downloadAndProcessRenderable(final Callable<InputStream> callable) {
        CompletableFuture<T> thenApplyAsync = CompletableFuture.supplyAsync(new Supplier() { // from class: c.d.b.a.q.k
            @Override // java.util.function.Supplier
            public final Object get() {
                return LoadRenderableFromSfbTask.this.a(callable);
            }
        }, ThreadPools.getThreadPoolExecutor()).thenComposeAsync(new Function() { // from class: c.d.b.a.q.l
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                return LoadRenderableFromSfbTask.this.b((SceneformBundleDef) obj);
            }
        }, ThreadPools.getMainExecutor()).thenApplyAsync(new Function() { // from class: c.d.b.a.q.q
            @Override // java.util.function.Function
            public final Object apply(Object obj) {
                return LoadRenderableFromSfbTask.this.c((SceneformBundleDef) obj);
            }
        }, ThreadPools.getMainExecutor());
        thenApplyAsync.exceptionally((Function<Throwable, ? extends T>) r.f4351a);
        return thenApplyAsync;
    }
}