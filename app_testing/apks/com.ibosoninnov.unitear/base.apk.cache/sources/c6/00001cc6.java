package com.google.ar.sceneform.rendering;

import com.google.android.filament.Camera;
import com.google.android.filament.Engine;
import com.google.android.filament.Fence;
import com.google.android.filament.IndexBuffer;
import com.google.android.filament.IndirectLight;
import com.google.android.filament.LightManager;
import com.google.android.filament.MaterialInstance;
import com.google.android.filament.NativeSurface;
import com.google.android.filament.RenderableManager;
import com.google.android.filament.Scene;
import com.google.android.filament.Skybox;
import com.google.android.filament.Stream;
import com.google.android.filament.SwapChain;
import com.google.android.filament.TransformManager;
import com.google.android.filament.VertexBuffer;
import com.google.android.filament.View;

/* loaded from: classes.dex */
public class FilamentEngineWrapper implements IEngine {
    public final Engine engine;

    public FilamentEngineWrapper(Engine engine) {
        this.engine = engine;
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public Camera createCamera() {
        return this.engine.createCamera();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public Fence createFence() {
        return this.engine.createFence();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public com.google.android.filament.Renderer createRenderer() {
        return this.engine.createRenderer();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public Scene createScene() {
        return this.engine.createScene();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public SwapChain createSwapChain(Object obj) {
        return this.engine.createSwapChain(obj);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public SwapChain createSwapChainFromNativeSurface(NativeSurface nativeSurface, long j) {
        return this.engine.createSwapChainFromNativeSurface(nativeSurface, j);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public View createView() {
        return this.engine.createView();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroy() {
        this.engine.destroy();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyCamera(Camera camera) {
        this.engine.destroyCamera(camera);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyEntity(int i) {
        this.engine.destroyEntity(i);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyFence(Fence fence) {
        this.engine.destroyFence(fence);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyIndexBuffer(IndexBuffer indexBuffer) {
        this.engine.destroyIndexBuffer(indexBuffer);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyIndirectLight(IndirectLight indirectLight) {
        this.engine.destroyIndirectLight(indirectLight);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyMaterial(com.google.android.filament.Material material) {
        this.engine.destroyMaterial(material);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyMaterialInstance(MaterialInstance materialInstance) {
        this.engine.destroyMaterialInstance(materialInstance);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyRenderer(com.google.android.filament.Renderer renderer) {
        this.engine.destroyRenderer(renderer);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyScene(Scene scene) {
        this.engine.destroyScene(scene);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroySkybox(Skybox skybox) {
        this.engine.destroySkybox(skybox);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyStream(Stream stream) {
        this.engine.destroyStream(stream);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroySwapChain(SwapChain swapChain) {
        this.engine.destroySwapChain(swapChain);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyTexture(com.google.android.filament.Texture texture) {
        this.engine.destroyTexture(texture);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyVertexBuffer(VertexBuffer vertexBuffer) {
        this.engine.destroyVertexBuffer(vertexBuffer);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void destroyView(View view) {
        this.engine.destroyView(view);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public void flushAndWait() {
        this.engine.flushAndWait();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public Engine getFilamentEngine() {
        return this.engine;
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public LightManager getLightManager() {
        return this.engine.getLightManager();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public RenderableManager getRenderableManager() {
        return this.engine.getRenderableManager();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public TransformManager getTransformManager() {
        return this.engine.getTransformManager();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public boolean isValid() {
        return this.engine.isValid();
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public Camera createCamera(int i) {
        return this.engine.createCamera(i);
    }

    @Override // com.google.ar.sceneform.rendering.IEngine
    public SwapChain createSwapChain(Object obj, long j) {
        return this.engine.createSwapChain(obj, j);
    }
}