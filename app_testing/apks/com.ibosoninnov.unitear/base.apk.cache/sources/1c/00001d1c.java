package com.google.ar.sceneform.rendering;

import android.util.Log;
import c.b.a.a.a;
import c.d.b.a.q.g0;
import com.google.android.filament.EntityManager;
import com.google.android.filament.IndexBuffer;
import com.google.android.filament.RenderableManager;
import com.google.android.filament.Scene;
import com.google.android.filament.VertexBuffer;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.RenderingResources;
import com.google.ar.sceneform.rendering.SimpleCameraStream;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.Preconditions;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.function.Consumer;
import java.util.function.Function;

/* loaded from: classes.dex */
public class SimpleCameraStream {
    private static final int FLOAT_SIZE_IN_BYTES = 4;
    public static final String MATERIAL_CAMERA_TEXTURE = "cameraTexture";
    private static final int POSITION_BUFFER_INDEX = 0;
    private static final String TAG = "SimpleCameraStream";
    private static final int UNINITIALIZED_FILAMENT_RENDERABLE = -1;
    private static final int UV_BUFFER_INDEX = 1;
    private static final int VERTEX_COUNT = 3;
    private final IndexBuffer cameraIndexBuffer;
    private ExternalTexture cameraTexture;
    private final int cameraTextureId;
    private final FloatBuffer cameraUvCoords;
    private final VertexBuffer cameraVertexBuffer;
    private final Scene scene;
    private final FloatBuffer transformedCameraUvCoords;
    private static final short[] CAMERA_INDICES = {0, 1, 2};
    private static float aspectCorrector = 1.66f;
    private static final float[] CAMERA_VERTICES = {1.66f * 1.0f, 1.0f, 1.0f, 1.66f * (-3.0f), 1.0f, 1.0f, 1.66f * 1.0f, -3.0f, 1.0f};
    private static final float[] CAMERA_UVS = {StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 2.0f, 2.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD};
    private int cameraStreamRenderable = -1;
    private Material defaultCameraMaterial = null;
    private Material cameraMaterial = null;
    private int renderablePriority = 7;
    private boolean isTextureInitialized = false;

    /* loaded from: classes.dex */
    public static final class CleanupCallback implements Runnable {
        private final IndexBuffer cameraIndexBuffer;
        private final int cameraStreamRenderable;
        private final VertexBuffer cameraVertexBuffer;
        private final Scene scene;

        public CleanupCallback(Scene scene, int i, IndexBuffer indexBuffer, VertexBuffer vertexBuffer) {
            this.scene = scene;
            this.cameraStreamRenderable = i;
            this.cameraIndexBuffer = indexBuffer;
            this.cameraVertexBuffer = vertexBuffer;
        }

        @Override // java.lang.Runnable
        public void run() {
            AndroidPreconditions.checkUiThread();
            IEngine engine = EngineInstance.getEngine();
            if (engine != null || engine.isValid()) {
                int i = this.cameraStreamRenderable;
                if (i != -1) {
                    this.scene.remove(i);
                }
                engine.destroyIndexBuffer(this.cameraIndexBuffer);
                engine.destroyVertexBuffer(this.cameraVertexBuffer);
            }
        }
    }

    public SimpleCameraStream(int i, Renderer renderer) {
        this.scene = renderer.getFilamentScene();
        this.cameraTextureId = i;
        IEngine engine = EngineInstance.getEngine();
        short[] sArr = CAMERA_INDICES;
        ShortBuffer allocate = ShortBuffer.allocate(sArr.length);
        allocate.put(sArr);
        IndexBuffer build = new IndexBuffer.Builder().indexCount(allocate.capacity()).bufferType(IndexBuffer.Builder.IndexType.USHORT).build(engine.getFilamentEngine());
        this.cameraIndexBuffer = build;
        allocate.rewind();
        ((IndexBuffer) Preconditions.checkNotNull(build)).setBuffer(engine.getFilamentEngine(), allocate);
        this.cameraUvCoords = createCameraUVBuffer();
        FloatBuffer createCameraUVBuffer = createCameraUVBuffer();
        this.transformedCameraUvCoords = createCameraUVBuffer;
        float[] fArr = CAMERA_VERTICES;
        FloatBuffer allocate2 = FloatBuffer.allocate(fArr.length);
        allocate2.put(fArr);
        VertexBuffer build2 = new VertexBuffer.Builder().vertexCount(3).bufferCount(2).attribute(VertexBuffer.VertexAttribute.POSITION, 0, VertexBuffer.AttributeType.FLOAT3, 0, (fArr.length / 3) * 4).attribute(VertexBuffer.VertexAttribute.UV0, 1, VertexBuffer.AttributeType.FLOAT2, 0, (CAMERA_UVS.length / 3) * 4).build(engine.getFilamentEngine());
        this.cameraVertexBuffer = build2;
        allocate2.rewind();
        ((VertexBuffer) Preconditions.checkNotNull(build2)).setBufferAt(engine.getFilamentEngine(), 0, allocate2);
        adjustCameraUvsForOpenGL();
        build2.setBufferAt(engine.getFilamentEngine(), 1, createCameraUVBuffer);
        Material.builder().setSource(renderer.getContext(), RenderingResources.GetSceneformResource(renderer.getContext(), RenderingResources.Resource.CAMERA_MATERIAL)).build().thenAccept(new Consumer() { // from class: c.d.b.a.q.h0
            @Override // java.util.function.Consumer
            public final void accept(Object obj) {
                SimpleCameraStream.this.a((Material) obj);
            }
        }).exceptionally((Function<Throwable, ? extends Void>) g0.f4328a);
    }

    private void adjustCameraUvsForOpenGL() {
        for (int i = 1; i < 6; i += 2) {
            FloatBuffer floatBuffer = this.transformedCameraUvCoords;
            floatBuffer.put(i, 1.0f - floatBuffer.get(i));
            String str = TAG;
            StringBuilder y = a.y("adjustCameraUvsForOpenGL ", i, " = ");
            y.append(1.0f - this.transformedCameraUvCoords.get(i));
            Log.d(str, y.toString());
        }
        this.transformedCameraUvCoords.rewind();
        try {
            Log.d("testCameraStream", "cameraUvCoords " + this.transformedCameraUvCoords.get(0) + ", " + this.transformedCameraUvCoords.get(1) + ", " + this.transformedCameraUvCoords.get(2) + ", " + this.transformedCameraUvCoords.get(3) + ", " + this.transformedCameraUvCoords.get(4) + ", " + this.transformedCameraUvCoords.get(5));
        } catch (Exception e2) {
            StringBuilder x = a.x("cameraUvCoords ");
            x.append(e2.toString());
            Log.d("testCameraStream", x.toString());
        }
    }

    private static FloatBuffer createCameraUVBuffer() {
        float[] fArr = CAMERA_UVS;
        FloatBuffer asFloatBuffer = ByteBuffer.allocateDirect(fArr.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        asFloatBuffer.put(fArr);
        asFloatBuffer.rewind();
        return asFloatBuffer;
    }

    private void initializeFilamentRenderable() {
        this.cameraStreamRenderable = EntityManager.get().create();
        new RenderableManager.Builder(1).castShadows(false).receiveShadows(false).culling(false).priority(this.renderablePriority).geometry(0, RenderableManager.PrimitiveType.TRIANGLES, this.cameraVertexBuffer, this.cameraIndexBuffer).material(0, ((Material) Preconditions.checkNotNull(this.cameraMaterial)).getFilamentMaterialInstance()).build(EngineInstance.getEngine().getFilamentEngine(), this.cameraStreamRenderable);
        this.scene.addEntity(this.cameraStreamRenderable);
        ResourceManager.getInstance().getSimpleCameraStreamCleanupRegistry().register(this, new CleanupCallback(this.scene, this.cameraStreamRenderable, this.cameraIndexBuffer, this.cameraVertexBuffer));
    }

    public /* synthetic */ void a(Material material) {
        this.defaultCameraMaterial = material;
        if (this.cameraMaterial == null) {
            setCameraMaterial(material);
        }
    }

    public int getRenderPriority() {
        return this.renderablePriority;
    }

    public void initializeTexture(int i, int i2) {
        if (isTextureInitialized()) {
            return;
        }
        this.cameraTexture = new ExternalTexture(this.cameraTextureId, i, i2);
        this.isTextureInitialized = true;
        Material material = this.cameraMaterial;
        if (material != null) {
            setCameraMaterial(material);
        }
    }

    public boolean isTextureInitialized() {
        return this.isTextureInitialized;
    }

    public void setCameraMaterial(Material material) {
        this.cameraMaterial = material;
        if (isTextureInitialized()) {
            material.setExternalTexture("cameraTexture", (ExternalTexture) Preconditions.checkNotNull(this.cameraTexture));
            if (this.cameraStreamRenderable == -1) {
                initializeFilamentRenderable();
                return;
            }
            RenderableManager renderableManager = EngineInstance.getEngine().getRenderableManager();
            renderableManager.setMaterialInstanceAt(renderableManager.getInstance(this.cameraStreamRenderable), 0, material.getFilamentMaterialInstance());
        }
    }

    public void setCameraMaterialToDefault() {
        Material material = this.defaultCameraMaterial;
        if (material != null) {
            setCameraMaterial(material);
        } else {
            this.cameraMaterial = null;
        }
    }

    public void setRenderPriority(int i) {
        this.renderablePriority = i;
        if (this.cameraStreamRenderable != -1) {
            RenderableManager renderableManager = EngineInstance.getEngine().getRenderableManager();
            renderableManager.setPriority(renderableManager.getInstance(this.cameraStreamRenderable), this.renderablePriority);
        }
    }

    public void initializeTexture(ExternalTexture externalTexture) {
        if (isTextureInitialized()) {
            return;
        }
        Log.d(TAG, "initializeTexture ----- external texture set");
        this.cameraTexture = externalTexture;
        this.isTextureInitialized = true;
        Material material = this.cameraMaterial;
        if (material != null) {
            setCameraMaterial(material);
        }
    }
}