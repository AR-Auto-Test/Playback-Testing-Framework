package com.google.ar.sceneform.rendering;

import android.util.Log;
import com.google.android.filament.Box;
import com.google.android.filament.Entity;
import com.google.android.filament.IndexBuffer;
import com.google.android.filament.RenderableManager;
import com.google.android.filament.VertexBuffer;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.RenderableInternalData;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/* loaded from: classes.dex */
public class RenderableInternalData implements IRenderableInternalData {
    private static final String TAG = "RenderableInternalData";
    private IndexBuffer indexBuffer;
    private FloatBuffer rawColorBuffer;
    private IntBuffer rawIndexBuffer;
    private FloatBuffer rawPositionBuffer;
    private FloatBuffer rawTangentsBuffer;
    private FloatBuffer rawUvBuffer;
    private VertexBuffer vertexBuffer;
    private final Vector3 centerAabb = Vector3.zero();
    private final Vector3 extentsAabb = Vector3.zero();
    private float transformScale = 1.0f;
    private final Vector3 transformOffset = Vector3.zero();
    private final ArrayList<MeshData> meshes = new ArrayList<>();

    /* loaded from: classes.dex */
    public static class MeshData {
        public int indexEnd;
        public int indexStart;
    }

    private void setupSkeleton(RenderableManager.Builder builder) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void buildInstanceData(Renderable renderable, @Entity int i) {
        IRenderableInternalData renderableData = renderable.getRenderableData();
        ArrayList<Material> materialBindings = renderable.getMaterialBindings();
        RenderableManager renderableManager = EngineInstance.getEngine().getRenderableManager();
        int renderableManager2 = renderableManager.getInstance(i);
        int size = renderableData.getMeshes().size();
        if (renderableManager2 != 0 && renderableManager.getPrimitiveCount(renderableManager2) == size) {
            renderableManager.setPriority(renderableManager2, renderable.getRenderPriority());
            renderableManager.setCastShadows(renderableManager2, renderable.isShadowCaster());
            renderableManager.setReceiveShadows(renderableManager2, renderable.isShadowReceiver());
        } else {
            if (renderableManager2 != 0) {
                renderableManager.destroy(i);
            }
            RenderableManager.Builder receiveShadows = new RenderableManager.Builder(size).priority(renderable.getRenderPriority()).castShadows(renderable.isShadowCaster()).receiveShadows(renderable.isShadowReceiver());
            setupSkeleton(receiveShadows);
            receiveShadows.build(EngineInstance.getEngine().getFilamentEngine(), i);
            renderableManager2 = renderableManager.getInstance(i);
            if (renderableManager2 == 0) {
                throw new AssertionError("Unable to create RenderableInstance.");
            }
        }
        int i2 = renderableManager2;
        Vector3 extentsAabb = renderableData.getExtentsAabb();
        Vector3 centerAabb = renderableData.getCenterAabb();
        renderableManager.setAxisAlignedBoundingBox(i2, new Box(centerAabb.x, centerAabb.y, centerAabb.z, extentsAabb.x, extentsAabb.y, extentsAabb.z));
        if (materialBindings.size() == size) {
            RenderableManager.PrimitiveType primitiveType = RenderableManager.PrimitiveType.TRIANGLES;
            for (int i3 = 0; i3 < size; i3++) {
                MeshData meshData = renderableData.getMeshes().get(i3);
                VertexBuffer vertexBuffer = renderableData.getVertexBuffer();
                IndexBuffer indexBuffer = renderableData.getIndexBuffer();
                if (vertexBuffer != null && indexBuffer != null) {
                    int i4 = meshData.indexStart;
                    renderableManager.setGeometryAt(i2, i3, primitiveType, vertexBuffer, indexBuffer, i4, meshData.indexEnd - i4);
                    renderableManager.setMaterialInstanceAt(i2, i3, materialBindings.get(i3).getFilamentMaterialInstance());
                } else {
                    throw new AssertionError("Internal Error: Failed to get vertex or index buffer");
                }
            }
            return;
        }
        throw new AssertionError("Material Bindings are out of sync with meshes.");
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void dispose() {
        AndroidPreconditions.checkUiThread();
        IEngine engine = EngineInstance.getEngine();
        if (engine == null || !engine.isValid()) {
            return;
        }
        VertexBuffer vertexBuffer = this.vertexBuffer;
        if (vertexBuffer != null) {
            engine.destroyVertexBuffer(vertexBuffer);
            this.vertexBuffer = null;
        }
        IndexBuffer indexBuffer = this.indexBuffer;
        if (indexBuffer != null) {
            engine.destroyIndexBuffer(indexBuffer);
            this.indexBuffer = null;
        }
    }

    public void finalize() {
        try {
            try {
                ThreadPools.getMainExecutor().execute(new Runnable() { // from class: c.d.b.a.q.f0
                    @Override // java.lang.Runnable
                    public final void run() {
                        RenderableInternalData.this.dispose();
                    }
                });
            } catch (Exception e2) {
                Log.e(TAG, "Error while Finalizing Renderable Internal Data.", e2);
            }
        } finally {
            super.finalize();
        }
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public List<String> getAnimationNames() {
        return Collections.emptyList();
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public Vector3 getCenterAabb() {
        return new Vector3(this.centerAabb);
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public Vector3 getExtentsAabb() {
        return new Vector3(this.extentsAabb);
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public IndexBuffer getIndexBuffer() {
        return this.indexBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public ArrayList<MeshData> getMeshes() {
        return this.meshes;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public FloatBuffer getRawColorBuffer() {
        return this.rawColorBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public IntBuffer getRawIndexBuffer() {
        return this.rawIndexBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public FloatBuffer getRawPositionBuffer() {
        return this.rawPositionBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public FloatBuffer getRawTangentsBuffer() {
        return this.rawTangentsBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public FloatBuffer getRawUvBuffer() {
        return this.rawUvBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public Vector3 getSizeAabb() {
        return this.extentsAabb.scaled(2.0f);
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public Vector3 getTransformOffset() {
        return new Vector3(this.transformOffset);
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public float getTransformScale() {
        return this.transformScale;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public VertexBuffer getVertexBuffer() {
        return this.vertexBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setAnimationNames(List<String> list) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setCenterAabb(Vector3 vector3) {
        this.centerAabb.set(vector3);
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setExtentsAabb(Vector3 vector3) {
        this.extentsAabb.set(vector3);
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setIndexBuffer(IndexBuffer indexBuffer) {
        this.indexBuffer = indexBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setRawColorBuffer(FloatBuffer floatBuffer) {
        this.rawColorBuffer = floatBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setRawIndexBuffer(IntBuffer intBuffer) {
        this.rawIndexBuffer = intBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setRawPositionBuffer(FloatBuffer floatBuffer) {
        this.rawPositionBuffer = floatBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setRawTangentsBuffer(FloatBuffer floatBuffer) {
        this.rawTangentsBuffer = floatBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setRawUvBuffer(FloatBuffer floatBuffer) {
        this.rawUvBuffer = floatBuffer;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setTransformOffset(Vector3 vector3) {
        this.transformOffset.set(vector3);
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setTransformScale(float f2) {
        this.transformScale = f2;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setVertexBuffer(VertexBuffer vertexBuffer) {
        this.vertexBuffer = vertexBuffer;
    }
}