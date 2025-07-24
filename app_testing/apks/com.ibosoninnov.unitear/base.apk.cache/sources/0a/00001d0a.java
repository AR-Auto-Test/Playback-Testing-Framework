package com.google.ar.sceneform.rendering;

import android.net.Uri;
import android.text.TextUtils;
import android.util.Log;
import com.google.android.filament.Box;
import com.google.android.filament.Entity;
import com.google.android.filament.EntityManager;
import com.google.android.filament.MaterialInstance;
import com.google.android.filament.RenderableManager;
import com.google.android.filament.TransformManager;
import com.google.android.filament.gltfio.Animator;
import com.google.android.filament.gltfio.AssetLoader;
import com.google.android.filament.gltfio.FilamentAsset;
import com.google.ar.sceneform.animation.AnimatableModel;
import com.google.ar.sceneform.animation.ModelAnimation;
import com.google.ar.sceneform.common.TransformProvider;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.ChangeId;
import com.google.ar.sceneform.utilities.LoadHelper;
import com.google.ar.sceneform.utilities.Preconditions;
import com.google.ar.sceneform.utilities.SceneformBufferUtils;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.function.Function;

/* loaded from: classes.dex */
public class RenderableInstance implements AnimatableModel {
    private static final String TAG = "RenderableInstance";
    public Renderer attachedRenderer;
    private Matrix cachedRelativeTransform;
    private Matrix cachedRelativeTransformInverse;
    @Entity
    private int childEntity;
    @Entity
    private int entity;
    public Animator filamentAnimator;
    public FilamentAsset filamentAsset;
    private ArrayList<Material> materialBindings;
    private ArrayList<String> materialNames;
    private final Renderable renderable;
    private SkinningModifier skinningModifier;
    private final TransformProvider transformProvider;
    public int renderableId = 0;
    private ArrayList<ModelAnimation> animations = new ArrayList<>();
    private int renderPriority = 4;
    private boolean isShadowCaster = true;
    private boolean isShadowReceiver = true;

    /* loaded from: classes.dex */
    public static final class CleanupCallback implements Runnable {
        private final int childEntity;
        private final int entity;

        public CleanupCallback(int i, int i2) {
            this.childEntity = i;
            this.entity = i2;
        }

        @Override // java.lang.Runnable
        public void run() {
            AndroidPreconditions.checkUiThread();
            IEngine engine = EngineInstance.getEngine();
            if (engine == null || !engine.isValid()) {
                return;
            }
            RenderableManager renderableManager = engine.getRenderableManager();
            int i = this.childEntity;
            if (i != 0) {
                renderableManager.destroy(i);
            }
            int i2 = this.entity;
            if (i2 != 0) {
                renderableManager.destroy(i2);
            }
        }
    }

    /* loaded from: classes.dex */
    public interface SkinningModifier {
        boolean isModifiedSinceLastRender();

        FloatBuffer modifyMaterialBoneTransformsBuffer(FloatBuffer floatBuffer);
    }

    public RenderableInstance(TransformProvider transformProvider, Renderable renderable) {
        this.entity = 0;
        this.childEntity = 0;
        Preconditions.checkNotNull(transformProvider, "Parameter \"transformProvider\" was null.");
        Preconditions.checkNotNull(renderable, "Parameter \"renderable\" was null.");
        this.transformProvider = transformProvider;
        this.renderable = renderable;
        this.materialBindings = new ArrayList<>(renderable.getMaterialBindings());
        this.materialNames = new ArrayList<>(renderable.getMaterialNames());
        this.entity = createFilamentEntity(EngineInstance.getEngine());
        Matrix relativeTransform = getRelativeTransform();
        if (relativeTransform != null) {
            this.childEntity = createFilamentChildEntity(EngineInstance.getEngine(), this.entity, relativeTransform);
        }
        createGltfModelInstance();
        createFilamentAssetModelInstance();
        ResourceManager.getInstance().getRenderableInstanceCleanupRegistry().register(this, new CleanupCallback(this.entity, this.childEntity));
    }

    private void attachFilamentAssetToRenderer() {
        FilamentAsset filamentAsset = this.filamentAsset;
        if (filamentAsset != null) {
            int[] entities = filamentAsset.getEntities();
            ((Renderer) Preconditions.checkNotNull(this.attachedRenderer)).getFilamentScene().addEntity(filamentAsset.getRoot());
            ((Renderer) Preconditions.checkNotNull(this.attachedRenderer)).getFilamentScene().addEntities(filamentAsset.getEntities());
            ((Renderer) Preconditions.checkNotNull(this.attachedRenderer)).getFilamentScene().addEntities(entities);
        }
    }

    @Entity
    private static int createFilamentChildEntity(IEngine iEngine, @Entity int i, Matrix matrix) {
        int create = EntityManager.get().create();
        TransformManager transformManager = iEngine.getTransformManager();
        transformManager.create(create, transformManager.getInstance(i), matrix.data);
        return create;
    }

    @Entity
    private static int createFilamentEntity(IEngine iEngine) {
        int create = EntityManager.get().create();
        iEngine.getTransformManager().create(create);
        return create;
    }

    private void setupSkeleton(IRenderableInternalData iRenderableInternalData) {
    }

    private void updateSkinning() {
        if (getFilamentAnimator() != null) {
            getFilamentAnimator().updateBoneMatrices();
        }
    }

    @Override // com.google.ar.sceneform.animation.AnimatableModel
    public boolean applyAnimationChange(ModelAnimation modelAnimation) {
        return false;
    }

    public void attachToRenderer(Renderer renderer) {
        renderer.addInstance(this);
        this.attachedRenderer = renderer;
        this.renderable.attachToRenderer(renderer);
        attachFilamentAssetToRenderer();
    }

    public void createFilamentAssetModelInstance() {
        String[] resourceUris;
        if (this.renderable.getRenderableData() instanceof RenderableInternalFilamentAssetData) {
            RenderableInternalFilamentAssetData renderableInternalFilamentAssetData = (RenderableInternalFilamentAssetData) this.renderable.getRenderableData();
            AssetLoader assetLoader = new AssetLoader(EngineInstance.getEngine().getFilamentEngine(), RenderableInternalFilamentAssetData.getMaterialProvider(), EntityManager.get());
            FilamentAsset createAssetFromBinary = renderableInternalFilamentAssetData.isGltfBinary ? assetLoader.createAssetFromBinary(renderableInternalFilamentAssetData.gltfByteBuffer) : assetLoader.createAssetFromJson(renderableInternalFilamentAssetData.gltfByteBuffer);
            if (createAssetFromBinary != null) {
                if (this.renderable.collisionShape == null) {
                    Box boundingBox = createAssetFromBinary.getBoundingBox();
                    float[] halfExtent = boundingBox.getHalfExtent();
                    float[] center = boundingBox.getCenter();
                    this.renderable.collisionShape = new com.google.ar.sceneform.collision.Box(new Vector3(halfExtent[0], halfExtent[1], halfExtent[2]).scaled(2.0f), new Vector3(center[0], center[1], center[2]));
                }
                Function<String, Uri> function = renderableInternalFilamentAssetData.urlResolver;
                for (String str : createAssetFromBinary.getResourceUris()) {
                    if (function == null) {
                        Log.e(TAG, "Failed to download uri " + str + " no url resolver.");
                    } else {
                        Uri apply = function.apply(str);
                        try {
                            renderableInternalFilamentAssetData.resourceLoader.addResourceData(str, ByteBuffer.wrap(SceneformBufferUtils.inputStreamCallableToByteArray(LoadHelper.fromUri(renderableInternalFilamentAssetData.context, apply))));
                        } catch (Exception e2) {
                            Log.e(TAG, "Failed to download data uri " + apply, e2);
                        }
                    }
                }
                renderableInternalFilamentAssetData.resourceLoader.loadResources(createAssetFromBinary);
                RenderableManager renderableManager = EngineInstance.getEngine().getRenderableManager();
                this.materialBindings.clear();
                this.materialNames.clear();
                for (int i : createAssetFromBinary.getEntities()) {
                    int renderableManager2 = renderableManager.getInstance(i);
                    if (renderableManager2 != 0) {
                        MaterialInstance materialInstanceAt = renderableManager.getMaterialInstanceAt(renderableManager2, 0);
                        this.materialNames.add(materialInstanceAt.getName());
                        Material material = new Material(new MaterialInternalDataGltfImpl(materialInstanceAt.getMaterial()));
                        material.updateGltfMaterialInstance(materialInstanceAt);
                        this.materialBindings.add(material);
                    }
                }
                TransformManager transformManager = EngineInstance.getEngine().getTransformManager();
                int transformManager2 = transformManager.getInstance(createAssetFromBinary.getRoot());
                int i2 = this.childEntity;
                if (i2 == 0) {
                    i2 = this.entity;
                }
                transformManager.setParent(transformManager2, transformManager.getInstance(i2));
                this.filamentAsset = createAssetFromBinary;
                setRenderPriority(this.renderable.getRenderPriority());
                setShadowCaster(this.renderable.isShadowCaster());
                setShadowReceiver(this.renderable.isShadowReceiver());
                this.filamentAnimator = createAssetFromBinary.getAnimator();
                this.animations = new ArrayList<>();
                for (int i3 = 0; i3 < this.filamentAnimator.getAnimationCount(); i3++) {
                    this.animations.add(new ModelAnimation(this, this.filamentAnimator.getAnimationName(i3), i3, this.filamentAnimator.getAnimationDuration(i3), getRenderable().getAnimationFrameRate()));
                }
                return;
            }
            throw new IllegalStateException("Failed to load gltf");
        }
    }

    public void createGltfModelInstance() {
    }

    public void destroy() {
        detachFromRenderer();
        if (this.renderable.getRenderableData() instanceof RenderableInternalFilamentAssetData) {
            RenderableInternalFilamentAssetData renderableInternalFilamentAssetData = (RenderableInternalFilamentAssetData) this.renderable.getRenderableData();
        }
    }

    public void detachFromRenderer() {
        if (this.attachedRenderer != null) {
            FilamentAsset filamentAsset = this.filamentAsset;
            if (filamentAsset != null) {
                for (int i : filamentAsset.getEntities()) {
                    this.attachedRenderer.getFilamentScene().removeEntity(i);
                }
                this.attachedRenderer.getFilamentScene().removeEntity(filamentAsset.getRoot());
            }
            this.attachedRenderer.removeInstance(this);
            this.renderable.detatchFromRenderer();
        }
    }

    @Override // com.google.ar.sceneform.animation.AnimatableModel
    public ModelAnimation getAnimation(int i) {
        Preconditions.checkElementIndex(i, getAnimationCount(), "No animation found at the given index");
        return this.animations.get(i);
    }

    @Override // com.google.ar.sceneform.animation.AnimatableModel
    public int getAnimationCount() {
        return this.animations.size();
    }

    @Entity
    public int getEntity() {
        return this.entity;
    }

    public Animator getFilamentAnimator() {
        return this.filamentAnimator;
    }

    public FilamentAsset getFilamentAsset() {
        return this.filamentAsset;
    }

    public Material getMaterial() {
        return getMaterial(0);
    }

    public ArrayList<Material> getMaterialBindings() {
        return this.materialBindings;
    }

    public String getMaterialName(int i) {
        Preconditions.checkState(this.materialNames.size() == this.materialBindings.size());
        if (i < 0 || i >= this.materialNames.size()) {
            return null;
        }
        return this.materialNames.get(i);
    }

    public ArrayList<String> getMaterialNames() {
        return this.materialNames;
    }

    public int getMaterialsCount() {
        return this.materialBindings.size();
    }

    public Matrix getRelativeTransform() {
        Matrix matrix = this.cachedRelativeTransform;
        if (matrix != null) {
            return matrix;
        }
        IRenderableInternalData renderableData = this.renderable.getRenderableData();
        float transformScale = renderableData.getTransformScale();
        Vector3 transformOffset = renderableData.getTransformOffset();
        if (transformScale == 1.0f && Vector3.equals(transformOffset, Vector3.zero())) {
            return null;
        }
        Matrix matrix2 = new Matrix();
        this.cachedRelativeTransform = matrix2;
        matrix2.makeScale(transformScale);
        this.cachedRelativeTransform.setTranslation(transformOffset);
        return this.cachedRelativeTransform;
    }

    public Matrix getRelativeTransformInverse() {
        Matrix matrix = this.cachedRelativeTransformInverse;
        if (matrix != null) {
            return matrix;
        }
        Matrix relativeTransform = getRelativeTransform();
        if (relativeTransform == null) {
            return null;
        }
        Matrix matrix2 = new Matrix();
        this.cachedRelativeTransformInverse = matrix2;
        Matrix.invert(relativeTransform, matrix2);
        return this.cachedRelativeTransformInverse;
    }

    public int getRenderPriority() {
        return this.renderPriority;
    }

    public Renderable getRenderable() {
        return this.renderable;
    }

    @Entity
    public int getRenderedEntity() {
        int i = this.childEntity;
        return i == 0 ? this.entity : i;
    }

    public Matrix getWorldModelMatrix() {
        return this.renderable.getFinalModelMatrix(this.transformProvider.getWorldModelMatrix());
    }

    public boolean isShadowCaster() {
        return this.isShadowCaster;
    }

    public boolean isShadowReceiver() {
        return this.isShadowReceiver;
    }

    public void prepareForDraw() {
        this.renderable.prepareForDraw();
        ChangeId id = this.renderable.getId();
        if (id.checkChanged(this.renderableId)) {
            IRenderableInternalData renderableData = this.renderable.getRenderableData();
            setupSkeleton(renderableData);
            renderableData.buildInstanceData(this.renderable, getRenderedEntity());
            this.renderableId = id.get();
            updateSkinning();
        } else if (updateAnimations(false)) {
            updateSkinning();
        }
    }

    public void setBlendOrderAt(int i, int i2) {
        RenderableManager renderableManager = EngineInstance.getEngine().getRenderableManager();
        renderableManager.setBlendOrderAt(renderableManager.getInstance(getRenderedEntity()), i, i2);
    }

    public void setCulling(boolean z) {
        RenderableManager renderableManager = EngineInstance.getEngine().getRenderableManager();
        int renderableManager2 = renderableManager.getInstance(getEntity());
        if (renderableManager2 != 0 && renderableManager.hasComponent(renderableManager2)) {
            renderableManager.setCulling(renderableManager2, this.isShadowCaster);
        }
        for (int i : getFilamentAsset().getEntities()) {
            int renderableManager3 = renderableManager.getInstance(i);
            if (renderableManager3 != 0) {
                renderableManager.setCulling(renderableManager3, z);
            }
        }
    }

    public void setMaterial(Material material) {
        setMaterial(0, material);
    }

    public void setModelMatrix(TransformManager transformManager, float[] fArr) {
        transformManager.setTransform(transformManager.getInstance(this.entity), fArr);
    }

    public void setRenderPriority(int i) {
        int[] entities = getFilamentAsset().getEntities();
        this.renderPriority = Math.min(7, Math.max(0, i));
        RenderableManager renderableManager = EngineInstance.getEngine().getRenderableManager();
        for (int i2 : entities) {
            int renderableManager2 = renderableManager.getInstance(i2);
            if (renderableManager2 != 0) {
                renderableManager.setPriority(renderableManager2, this.renderPriority);
            }
        }
    }

    public void setShadowCaster(boolean z) {
        this.isShadowCaster = z;
        RenderableManager renderableManager = EngineInstance.getEngine().getRenderableManager();
        int renderableManager2 = renderableManager.getInstance(getEntity());
        if (renderableManager2 != 0) {
            renderableManager.setCastShadows(renderableManager2, z);
        }
    }

    public void setShadowReceiver(boolean z) {
        this.isShadowReceiver = z;
        RenderableManager renderableManager = EngineInstance.getEngine().getRenderableManager();
        int renderableManager2 = renderableManager.getInstance(getEntity());
        if (renderableManager2 != 0) {
            renderableManager.setReceiveShadows(renderableManager2, z);
        }
    }

    public void setSkinningModifier(SkinningModifier skinningModifier) {
        this.skinningModifier = skinningModifier;
    }

    public boolean updateAnimations(boolean z) {
        boolean z2 = false;
        for (int i = 0; i < getAnimationCount(); i++) {
            ModelAnimation animation = getAnimation(i);
            if (z || animation.isDirty()) {
                if (getFilamentAnimator() != null) {
                    getFilamentAnimator().applyAnimation(i, animation.getTimePosition());
                }
                animation.setDirty(false);
                z2 = true;
            }
        }
        return z2;
    }

    public Material getMaterial(int i) {
        if (i < this.materialBindings.size()) {
            return this.materialBindings.get(i);
        }
        return null;
    }

    public void setMaterial(int i, Material material) {
        for (int i2 = 0; i2 < getFilamentAsset().getEntities().length; i2++) {
            setMaterial(i2, i, material);
        }
    }

    public Material getMaterial(String str) {
        for (int i = 0; i < this.materialBindings.size(); i++) {
            if (TextUtils.equals(this.materialNames.get(i), str)) {
                return this.materialBindings.get(i);
            }
        }
        return null;
    }

    public void setMaterial(int i, int i2, Material material) {
        int[] entities = getFilamentAsset().getEntities();
        Preconditions.checkElementIndex(i, entities.length, "No entity found at the given index");
        this.materialBindings.set(i, material);
        RenderableManager renderableManager = EngineInstance.getEngine().getRenderableManager();
        int renderableManager2 = renderableManager.getInstance(entities[i]);
        if (renderableManager2 != 0) {
            renderableManager.setMaterialInstanceAt(renderableManager2, i2, material.getFilamentMaterialInstance());
        }
    }
}