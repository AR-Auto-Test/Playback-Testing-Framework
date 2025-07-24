package com.google.ar.sceneform.rendering;

import android.content.Context;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.collision.Box;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.RenderViewToExternalTexture;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.RenderableDefinition;
import com.google.ar.sceneform.rendering.RenderingResources;
import com.google.ar.sceneform.rendering.Vertex;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.resources.ResourceRegistry;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.Preconditions;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.Consumer;
import java.util.function.Function;

/* loaded from: classes.dex */
public class ViewRenderable extends Renderable {
    private static final String TAG = "ViewRenderable";
    private HorizontalAlignment horizontalAlignment;
    private boolean isInitialized;
    private final RenderViewToExternalTexture.OnViewSizeChangedListener onViewSizeChangedListener;
    private Renderer renderer;
    private VerticalAlignment verticalAlignment;
    private final View view;
    private ViewRenderableInternalData viewRenderableData;
    private final Matrix viewScaleMatrix;
    private ViewSizer viewSizer;

    /* renamed from: com.google.ar.sceneform.rendering.ViewRenderable$1  reason: invalid class name */
    /* loaded from: classes.dex */
    public static /* synthetic */ class AnonymousClass1 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$HorizontalAlignment;
        public static final /* synthetic */ int[] $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$VerticalAlignment;

        static {
            VerticalAlignment.values();
            int[] iArr = new int[3];
            $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$VerticalAlignment = iArr;
            try {
                iArr[VerticalAlignment.BOTTOM.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$VerticalAlignment[VerticalAlignment.CENTER.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$VerticalAlignment[VerticalAlignment.TOP.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
            HorizontalAlignment.values();
            int[] iArr2 = new int[3];
            $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$HorizontalAlignment = iArr2;
            try {
                iArr2[HorizontalAlignment.LEFT.ordinal()] = 1;
            } catch (NoSuchFieldError unused4) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$HorizontalAlignment[HorizontalAlignment.CENTER.ordinal()] = 2;
            } catch (NoSuchFieldError unused5) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$HorizontalAlignment[HorizontalAlignment.RIGHT.ordinal()] = 3;
            } catch (NoSuchFieldError unused6) {
            }
        }
    }

    /* loaded from: classes.dex */
    public static final class Builder extends Renderable.Builder<ViewRenderable, Builder> {
        private static final int DEFAULT_DP_TO_METERS = 250;
        private HorizontalAlignment horizontalAlignment;
        private OptionalInt resourceId;
        private VerticalAlignment verticalAlignment;
        private View view;
        private ViewSizer viewSizer;

        public /* synthetic */ Builder(AnonymousClass1 anonymousClass1) {
            this();
        }

        private View inflateViewFromResourceId() {
            if (this.context != null) {
                return LayoutInflater.from(this.context).inflate(this.resourceId.getAsInt(), (ViewGroup) new FrameLayout(this.context), false);
            }
            throw new AssertionError("Context cannot be null");
        }

        public /* synthetic */ CompletionStage a(Void r1) {
            return super.build();
        }

        /* JADX DEBUG: Type inference failed for r0v9. Raw type applied. Possible types: java.util.concurrent.CompletableFuture<U>, java.util.concurrent.CompletableFuture<com.google.ar.sceneform.rendering.ViewRenderable> */
        @Override // com.google.ar.sceneform.rendering.Renderable.Builder
        public CompletableFuture<ViewRenderable> build() {
            if (!hasSource().booleanValue() && this.context != null) {
                this.registryId = this.view;
                Material.Builder builder = Material.builder();
                Context context = this.context;
                return builder.setSource(context, RenderingResources.GetSceneformResource(context, RenderingResources.Resource.VIEW_RENDERABLE_MATERIAL)).build().thenAccept(new Consumer() { // from class: c.d.b.a.q.n0
                    @Override // java.util.function.Consumer
                    public final void accept(Object obj) {
                        ViewRenderable.Builder builder2 = ViewRenderable.Builder.this;
                        Objects.requireNonNull(builder2);
                        ArrayList arrayList = new ArrayList();
                        arrayList.add(Vertex.builder().setPosition(new Vector3(-0.5f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)).setNormal(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f)).setUvCoordinate(new Vertex.UvCoordinate(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)).build());
                        arrayList.add(Vertex.builder().setPosition(new Vector3(0.5f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)).setNormal(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f)).setUvCoordinate(new Vertex.UvCoordinate(1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)).build());
                        arrayList.add(Vertex.builder().setPosition(new Vector3(-0.5f, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)).setNormal(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f)).setUvCoordinate(new Vertex.UvCoordinate(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f)).build());
                        arrayList.add(Vertex.builder().setPosition(new Vector3(0.5f, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)).setNormal(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f)).setUvCoordinate(new Vertex.UvCoordinate(1.0f, 1.0f)).build());
                        ArrayList arrayList2 = new ArrayList();
                        arrayList2.add(0);
                        arrayList2.add(1);
                        arrayList2.add(2);
                        arrayList2.add(1);
                        arrayList2.add(3);
                        arrayList2.add(2);
                        builder2.setSource(RenderableDefinition.builder().setVertices(arrayList).setSubmeshes(Arrays.asList(RenderableDefinition.Submesh.builder().setTriangleIndices(arrayList2).setMaterial((Material) obj).build())).build());
                    }
                }).thenCompose(new Function() { // from class: c.d.b.a.q.m0
                    @Override // java.util.function.Function
                    public final Object apply(Object obj) {
                        return ViewRenderable.Builder.this.a((Void) obj);
                    }
                });
            }
            return super.build();
        }

        @Override // com.google.ar.sceneform.rendering.Renderable.Builder
        public void checkPreconditions() {
            super.checkPreconditions();
            if (this.resourceId.isPresent() || this.view != null) {
                if (this.resourceId.isPresent() && this.view != null) {
                    throw new AssertionError("ViewRenderable must have a resourceId or a view as a source. This one has both.");
                }
                return;
            }
            throw new AssertionError("ViewRenderable must have a source.");
        }

        @Override // com.google.ar.sceneform.rendering.Renderable.Builder
        public Class<ViewRenderable> getRenderableClass() {
            return ViewRenderable.class;
        }

        @Override // com.google.ar.sceneform.rendering.Renderable.Builder
        public ResourceRegistry<ViewRenderable> getRenderableRegistry() {
            return ResourceManager.getInstance().getViewRenderableRegistry();
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // com.google.ar.sceneform.rendering.Renderable.Builder
        public Builder getSelf() {
            return this;
        }

        public Builder setHorizontalAlignment(HorizontalAlignment horizontalAlignment) {
            this.horizontalAlignment = horizontalAlignment;
            return this;
        }

        public Builder setSizer(ViewSizer viewSizer) {
            Preconditions.checkNotNull(viewSizer, "Parameter \"viewSizer\" was null.");
            this.viewSizer = viewSizer;
            return this;
        }

        public Builder setVerticalAlignment(VerticalAlignment verticalAlignment) {
            this.verticalAlignment = verticalAlignment;
            return this;
        }

        public Builder setView(Context context, View view) {
            this.view = view;
            this.context = context;
            this.registryId = view;
            return this;
        }

        private Builder() {
            this.viewSizer = new DpToMetersViewSizer(250);
            this.verticalAlignment = VerticalAlignment.BOTTOM;
            this.horizontalAlignment = HorizontalAlignment.CENTER;
            this.resourceId = OptionalInt.empty();
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // com.google.ar.sceneform.rendering.Renderable.Builder
        public ViewRenderable makeRenderable() {
            if (this.view != null) {
                return new ViewRenderable(this, this.view);
            }
            return new ViewRenderable(this, inflateViewFromResourceId());
        }

        public Builder setView(Context context, int i) {
            this.resourceId = OptionalInt.of(i);
            this.context = context;
            this.registryId = null;
            return this;
        }
    }

    /* loaded from: classes.dex */
    public enum HorizontalAlignment {
        LEFT,
        CENTER,
        RIGHT
    }

    /* loaded from: classes.dex */
    public enum VerticalAlignment {
        BOTTOM,
        CENTER,
        TOP
    }

    public ViewRenderable(Builder builder, View view) {
        super(builder);
        this.viewScaleMatrix = new Matrix();
        this.verticalAlignment = VerticalAlignment.BOTTOM;
        this.horizontalAlignment = HorizontalAlignment.CENTER;
        RenderViewToExternalTexture.OnViewSizeChangedListener onViewSizeChangedListener = new RenderViewToExternalTexture.OnViewSizeChangedListener() { // from class: c.d.b.a.q.p0
            @Override // com.google.ar.sceneform.rendering.RenderViewToExternalTexture.OnViewSizeChangedListener
            public final void onViewSizeChanged(int i, int i2) {
                ViewRenderable.this.b(i, i2);
            }
        };
        this.onViewSizeChangedListener = onViewSizeChangedListener;
        Preconditions.checkNotNull(view, "Parameter \"view\" was null.");
        this.view = view;
        this.viewSizer = builder.viewSizer;
        this.horizontalAlignment = builder.horizontalAlignment;
        this.verticalAlignment = builder.verticalAlignment;
        RenderViewToExternalTexture renderViewToExternalTexture = new RenderViewToExternalTexture(view.getContext(), view);
        renderViewToExternalTexture.addOnViewSizeChangedListener(onViewSizeChangedListener);
        ViewRenderableInternalData viewRenderableInternalData = new ViewRenderableInternalData(renderViewToExternalTexture);
        this.viewRenderableData = viewRenderableInternalData;
        viewRenderableInternalData.retain();
        this.collisionShape = new Box(Vector3.zero());
    }

    public static Builder builder() {
        AndroidPreconditions.checkMinAndroidApiLevel();
        return new Builder(null);
    }

    private float getOffsetRatioForAlignment(HorizontalAlignment horizontalAlignment) {
        IRenderableInternalData renderableData = getRenderableData();
        Vector3 centerAabb = renderableData.getCenterAabb();
        Vector3 extentsAabb = renderableData.getExtentsAabb();
        int ordinal = horizontalAlignment.ordinal();
        if (ordinal != 0) {
            if (ordinal != 1) {
                if (ordinal == 2) {
                    return (-centerAabb.x) - extentsAabb.x;
                }
                throw new IllegalStateException("Invalid HorizontalAlignment: " + horizontalAlignment);
            }
            return -centerAabb.x;
        }
        return (-centerAabb.x) + extentsAabb.x;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void updateSuggestedCollisionShape() {
        Box box;
        if (getId().isEmpty() || (box = (Box) this.collisionShape) == null) {
            return;
        }
        IRenderableInternalData renderableData = getRenderableData();
        Vector3 size = this.viewSizer.getSize(this.view);
        Vector3 sizeAabb = renderableData.getSizeAabb();
        sizeAabb.x *= size.x;
        sizeAabb.y *= size.y;
        Vector3 centerAabb = renderableData.getCenterAabb();
        float f2 = centerAabb.x * size.x;
        centerAabb.x = f2;
        centerAabb.y *= size.y;
        centerAabb.x = (getOffsetRatioForAlignment(this.horizontalAlignment) * sizeAabb.x) + f2;
        centerAabb.y = (getOffsetRatioForAlignment(this.verticalAlignment) * sizeAabb.y) + centerAabb.y;
        box.setSize(sizeAabb);
        box.setCenter(centerAabb);
    }

    private void updateSuggestedCollisionShapeAsync() {
        this.view.post(new Runnable() { // from class: c.d.b.a.q.o0
            @Override // java.lang.Runnable
            public final void run() {
                ViewRenderable.this.updateSuggestedCollisionShape();
            }
        });
    }

    @Override // com.google.ar.sceneform.rendering.Renderable
    public void attachToRenderer(Renderer renderer) {
        ((ViewRenderableInternalData) Preconditions.checkNotNull(this.viewRenderableData)).getRenderView().attachView(renderer.getViewAttachmentManager());
        this.renderer = renderer;
    }

    public /* synthetic */ void b(int i, int i2) {
        if (this.isInitialized) {
            updateSuggestedCollisionShapeAsync();
        }
    }

    @Override // com.google.ar.sceneform.rendering.Renderable
    public void detatchFromRenderer() {
        ((ViewRenderableInternalData) Preconditions.checkNotNull(this.viewRenderableData)).getRenderView().detachView();
        this.renderer = null;
    }

    public void dispose() {
        AndroidPreconditions.checkUiThread();
        ViewRenderableInternalData viewRenderableInternalData = this.viewRenderableData;
        if (viewRenderableInternalData != null) {
            viewRenderableInternalData.getRenderView().removeOnViewSizeChangedListener(this.onViewSizeChangedListener);
            viewRenderableInternalData.release();
            this.viewRenderableData = null;
        }
    }

    public void finalize() {
        try {
            try {
                ThreadPools.getMainExecutor().execute(new Runnable() { // from class: c.d.b.a.q.l0
                    @Override // java.lang.Runnable
                    public final void run() {
                        ViewRenderable.this.dispose();
                    }
                });
            } catch (Exception e2) {
                Log.e(TAG, "Error while Finalizing View Renderable.", e2);
            }
        } finally {
            super.finalize();
        }
    }

    @Override // com.google.ar.sceneform.rendering.Renderable
    public Matrix getFinalModelMatrix(Matrix matrix) {
        Preconditions.checkNotNull(matrix, "Parameter \"originalMatrix\" was null.");
        Vector3 size = this.viewSizer.getSize(this.view);
        this.viewScaleMatrix.makeScale(new Vector3(size.x, size.y, 1.0f));
        this.viewScaleMatrix.setTranslation(new Vector3(getOffsetRatioForAlignment(this.horizontalAlignment) * size.x, getOffsetRatioForAlignment(this.verticalAlignment) * size.y, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
        Matrix matrix2 = this.viewScaleMatrix;
        Matrix.multiply(matrix, matrix2, matrix2);
        return this.viewScaleMatrix;
    }

    public HorizontalAlignment getHorizontalAlignment() {
        return this.horizontalAlignment;
    }

    public ViewSizer getSizer() {
        return this.viewSizer;
    }

    public VerticalAlignment getVerticalAlignment() {
        return this.verticalAlignment;
    }

    public View getView() {
        return this.view;
    }

    @Override // com.google.ar.sceneform.rendering.Renderable
    public void prepareForDraw() {
        if (getId().isEmpty()) {
            return;
        }
        RenderViewToExternalTexture renderView = ((ViewRenderableInternalData) Preconditions.checkNotNull(this.viewRenderableData)).getRenderView();
        if (renderView.isAttachedToWindow() && renderView.isLaidOut() && renderView.hasDrawnToSurfaceTexture()) {
            if (!this.isInitialized) {
                getMaterial().setExternalTexture("viewTexture", renderView.getExternalTexture());
                updateSuggestedCollisionShape();
                this.isInitialized = true;
            }
            Renderer renderer = this.renderer;
            if (renderer != null && renderer.isFrontFaceWindingInverted()) {
                getMaterial().setFloat2("offsetUv", 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            }
            super.prepareForDraw();
        }
    }

    public void setHorizontalAlignment(HorizontalAlignment horizontalAlignment) {
        this.horizontalAlignment = horizontalAlignment;
        updateSuggestedCollisionShape();
    }

    public void setSizer(ViewSizer viewSizer) {
        Preconditions.checkNotNull(viewSizer, "Parameter \"viewSizer\" was null.");
        this.viewSizer = viewSizer;
        updateSuggestedCollisionShape();
    }

    public void setVerticalAlignment(VerticalAlignment verticalAlignment) {
        this.verticalAlignment = verticalAlignment;
        updateSuggestedCollisionShape();
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.rendering.Renderable
    public ViewRenderable makeCopy() {
        return new ViewRenderable(this);
    }

    private float getOffsetRatioForAlignment(VerticalAlignment verticalAlignment) {
        IRenderableInternalData renderableData = getRenderableData();
        Vector3 centerAabb = renderableData.getCenterAabb();
        Vector3 extentsAabb = renderableData.getExtentsAabb();
        int ordinal = verticalAlignment.ordinal();
        if (ordinal != 0) {
            if (ordinal != 1) {
                if (ordinal == 2) {
                    return (-centerAabb.y) - extentsAabb.y;
                }
                throw new IllegalStateException("Invalid VerticalAlignment: " + verticalAlignment);
            }
            return -centerAabb.y;
        }
        return (-centerAabb.y) + extentsAabb.y;
    }

    public ViewRenderable(ViewRenderable viewRenderable) {
        super(viewRenderable);
        this.viewScaleMatrix = new Matrix();
        this.verticalAlignment = VerticalAlignment.BOTTOM;
        this.horizontalAlignment = HorizontalAlignment.CENTER;
        RenderViewToExternalTexture.OnViewSizeChangedListener onViewSizeChangedListener = new RenderViewToExternalTexture.OnViewSizeChangedListener() { // from class: c.d.b.a.q.p0
            @Override // com.google.ar.sceneform.rendering.RenderViewToExternalTexture.OnViewSizeChangedListener
            public final void onViewSizeChanged(int i, int i2) {
                ViewRenderable.this.b(i, i2);
            }
        };
        this.onViewSizeChangedListener = onViewSizeChangedListener;
        this.view = viewRenderable.view;
        this.viewSizer = viewRenderable.viewSizer;
        this.horizontalAlignment = viewRenderable.horizontalAlignment;
        this.verticalAlignment = viewRenderable.verticalAlignment;
        ViewRenderableInternalData viewRenderableInternalData = (ViewRenderableInternalData) Preconditions.checkNotNull(viewRenderable.viewRenderableData);
        this.viewRenderableData = viewRenderableInternalData;
        viewRenderableInternalData.retain();
        this.viewRenderableData.getRenderView().addOnViewSizeChangedListener(onViewSizeChangedListener);
    }
}