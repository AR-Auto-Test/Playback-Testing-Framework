package com.google.ar.sceneform.rendering;

import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.resources.ResourceRegistry;
import com.google.ar.sceneform.utilities.AndroidPreconditions;

/* loaded from: classes.dex */
public class ModelRenderable extends Renderable {

    /* loaded from: classes.dex */
    public static final class Builder extends Renderable.Builder<ModelRenderable, Builder> {
        @Override // com.google.ar.sceneform.rendering.Renderable.Builder
        public Class<ModelRenderable> getRenderableClass() {
            return ModelRenderable.class;
        }

        @Override // com.google.ar.sceneform.rendering.Renderable.Builder
        public ResourceRegistry<ModelRenderable> getRenderableRegistry() {
            return ResourceManager.getInstance().getModelRenderableRegistry();
        }

        /* JADX DEBUG: Method merged with bridge method */
        /* JADX WARN: Can't rename method to resolve collision */
        @Override // com.google.ar.sceneform.rendering.Renderable.Builder
        public Builder getSelf() {
            return this;
        }

        /* JADX DEBUG: Method merged with bridge method */
        /* JADX WARN: Can't rename method to resolve collision */
        @Override // com.google.ar.sceneform.rendering.Renderable.Builder
        public ModelRenderable makeRenderable() {
            return new ModelRenderable(this);
        }
    }

    public static Builder builder() {
        AndroidPreconditions.checkMinAndroidApiLevel();
        return new Builder();
    }

    private void copyAnimationFrom(ModelRenderable modelRenderable) {
    }

    private ModelRenderable(Builder builder) {
        super(builder);
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.rendering.Renderable
    public ModelRenderable makeCopy() {
        return new ModelRenderable(this);
    }

    private ModelRenderable(ModelRenderable modelRenderable) {
        super(modelRenderable);
        copyAnimationFrom(modelRenderable);
    }
}