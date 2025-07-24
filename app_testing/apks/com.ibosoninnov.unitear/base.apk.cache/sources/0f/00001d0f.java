package com.google.ar.sceneform.rendering;

import android.content.Context;
import android.net.Uri;
import com.google.android.filament.IndexBuffer;
import com.google.android.filament.VertexBuffer;
import com.google.android.filament.gltfio.MaterialProvider;
import com.google.android.filament.gltfio.ResourceLoader;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.RenderableInternalData;
import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/* loaded from: classes.dex */
public class RenderableInternalFilamentAssetData implements IRenderableInternalData {
    public static MaterialProvider materialProvider;
    public Context context;
    public Buffer gltfByteBuffer;
    public boolean isGltfBinary;
    public ResourceLoader resourceLoader;
    public Function<String, Uri> urlResolver;

    public static MaterialProvider getMaterialProvider() {
        if (materialProvider == null) {
            materialProvider = new MaterialProvider(EngineInstance.getEngine().getFilamentEngine());
        }
        return materialProvider;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void buildInstanceData(Renderable renderable, int i) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void dispose() {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public List<String> getAnimationNames() {
        return new ArrayList();
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public Vector3 getCenterAabb() {
        return Vector3.zero();
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public Vector3 getExtentsAabb() {
        throw new IllegalStateException("Not Implemented");
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public IndexBuffer getIndexBuffer() {
        return null;
    }

    public ArrayList<Integer> getMaterialBindingIds() {
        return new ArrayList<>();
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public ArrayList<RenderableInternalData.MeshData> getMeshes() {
        return new ArrayList<>();
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public FloatBuffer getRawColorBuffer() {
        return null;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public IntBuffer getRawIndexBuffer() {
        return null;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public FloatBuffer getRawPositionBuffer() {
        return null;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public FloatBuffer getRawTangentsBuffer() {
        return null;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public FloatBuffer getRawUvBuffer() {
        return null;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public Vector3 getSizeAabb() {
        return Vector3.zero();
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public Vector3 getTransformOffset() {
        return Vector3.zero();
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public float getTransformScale() {
        return 1.0f;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public VertexBuffer getVertexBuffer() {
        return null;
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setAnimationNames(List<String> list) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setCenterAabb(Vector3 vector3) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setExtentsAabb(Vector3 vector3) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setIndexBuffer(IndexBuffer indexBuffer) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setRawColorBuffer(FloatBuffer floatBuffer) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setRawIndexBuffer(IntBuffer intBuffer) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setRawPositionBuffer(FloatBuffer floatBuffer) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setRawTangentsBuffer(FloatBuffer floatBuffer) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setRawUvBuffer(FloatBuffer floatBuffer) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setTransformOffset(Vector3 vector3) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setTransformScale(float f2) {
    }

    @Override // com.google.ar.sceneform.rendering.IRenderableInternalData
    public void setVertexBuffer(VertexBuffer vertexBuffer) {
    }
}