package com.google.ar.sceneform;

import android.view.MotionEvent;
import android.view.ViewConfiguration;
import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.collision.Collider;
import com.google.ar.sceneform.collision.CollisionShape;
import com.google.ar.sceneform.common.TransformProvider;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Light;
import com.google.ar.sceneform.rendering.LightInstance;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.RenderableInstance;
import com.google.ar.sceneform.rendering.Renderer;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.Preconditions;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Predicate;

/* loaded from: classes.dex */
public class Node extends NodeParent implements TransformProvider {
    private static final String DEFAULT_NAME = "Node";
    private static final int DEFAULT_TOUCH_SLOP = 8;
    private static final float DIRECTION_UP_EPSILON = 0.99f;
    private static final int LOCAL_DIRTY_FLAGS = 63;
    private static final int LOCAL_TRANSFORM_DIRTY = 1;
    private static final int WORLD_DIRTY_FLAGS = 62;
    private static final int WORLD_INVERSE_TRANSFORM_DIRTY = 4;
    private static final int WORLD_POSITION_DIRTY = 8;
    private static final int WORLD_ROTATION_DIRTY = 16;
    private static final int WORLD_SCALE_DIRTY = 32;
    private static final int WORLD_TRANSFORM_DIRTY = 2;
    private int accurateTouch;
    private boolean active;
    private boolean allowDispatchTransformChangedListeners;
    private final Matrix cachedLocalModelMatrix;
    private final Matrix cachedWorldModelMatrix;
    private final Matrix cachedWorldModelMatrixInverse;
    private final Vector3 cachedWorldPosition;
    private final Quaternion cachedWorldRotation;
    private final Vector3 cachedWorldScale;
    private Collider collider;
    private CollisionShape collisionShape;
    private int dirtyTransformFlags;
    private boolean enabled;
    private final ArrayList<LifecycleListener> lifecycleListeners;
    private LightInstance lightInstance;
    private final Vector3 localScale;
    private OnAccurateTapListner onAccurateTapListner;
    private OnTapListener onTapListener;
    private OnTouchListener onTouchListener;
    public NodeParent parent;
    private Node parentAsNode;
    private int renderableId;
    private RenderableInstance renderableInstance;
    private Scene scene;
    private TapTrackingData tapTrackingData;
    private final ArrayList<TransformChangedListener> transformChangedListeners;
    private String name = DEFAULT_NAME;
    private int nameHash = 2433570;
    private final Vector3 localPosition = new Vector3();
    private final Quaternion localRotation = new Quaternion();

    /* loaded from: classes.dex */
    public interface LifecycleListener {
        void onActivated(Node node);

        void onDeactivated(Node node);

        void onUpdated(Node node, FrameTime frameTime);
    }

    /* loaded from: classes.dex */
    public interface OnAccurateTapListner {
        void onAccurateTap(HitTestResult hitTestResult, MotionEvent motionEvent);
    }

    /* loaded from: classes.dex */
    public interface OnTapListener {
        void onTap(HitTestResult hitTestResult, MotionEvent motionEvent);
    }

    /* loaded from: classes.dex */
    public interface OnTouchListener {
        boolean onTouch(HitTestResult hitTestResult, MotionEvent motionEvent);
    }

    /* loaded from: classes.dex */
    public static class TapTrackingData {
        public final Node downNode;
        public final Vector3 downPosition;

        public TapTrackingData(Node node, Vector3 vector3) {
            this.downNode = node;
            this.downPosition = new Vector3(vector3);
        }
    }

    /* loaded from: classes.dex */
    public interface TransformChangedListener {
        void onTransformChanged(Node node, Node node2);
    }

    public Node() {
        Vector3 vector3 = new Vector3();
        this.localScale = vector3;
        this.cachedLocalModelMatrix = new Matrix();
        this.cachedWorldPosition = new Vector3();
        this.cachedWorldRotation = new Quaternion();
        Vector3 vector32 = new Vector3();
        this.cachedWorldScale = vector32;
        this.cachedWorldModelMatrix = new Matrix();
        this.cachedWorldModelMatrixInverse = new Matrix();
        this.dirtyTransformFlags = 63;
        this.enabled = true;
        this.active = false;
        this.renderableId = 0;
        this.lifecycleListeners = new ArrayList<>();
        this.transformChangedListeners = new ArrayList<>();
        this.allowDispatchTransformChangedListeners = true;
        this.tapTrackingData = null;
        AndroidPreconditions.checkUiThread();
        vector3.set(1.0f, 1.0f, 1.0f);
        vector32.set(vector3);
    }

    private void activate() {
        Scene scene;
        RenderableInstance renderableInstance;
        AndroidPreconditions.checkUiThread();
        if (!this.active) {
            this.active = true;
            Scene scene2 = this.scene;
            if (scene2 != null && !scene2.isUnderTesting() && (renderableInstance = this.renderableInstance) != null) {
                renderableInstance.attachToRenderer(getRendererOrDie());
            }
            LightInstance lightInstance = this.lightInstance;
            if (lightInstance != null) {
                lightInstance.attachToRenderer(getRendererOrDie());
            }
            Collider collider = this.collider;
            if (collider != null && (scene = this.scene) != null) {
                collider.setAttachedCollisionSystem(scene.collisionSystem);
            }
            onActivate();
            Iterator<LifecycleListener> it = this.lifecycleListeners.iterator();
            while (it.hasNext()) {
                it.next().onActivated(this);
            }
            return;
        }
        throw new AssertionError("Cannot call activate while already active.");
    }

    private void createLightInstance(Light light) {
        LightInstance createInstance = light.createInstance(this);
        this.lightInstance = createInstance;
        Objects.requireNonNull(createInstance, "light.createInstance() failed - which should not happen.");
        if (this.active) {
            createInstance.attachToRenderer(getRendererOrDie());
        }
    }

    private void deactivate() {
        AndroidPreconditions.checkUiThread();
        if (this.active) {
            this.active = false;
            RenderableInstance renderableInstance = this.renderableInstance;
            if (renderableInstance != null) {
                renderableInstance.detachFromRenderer();
            }
            LightInstance lightInstance = this.lightInstance;
            if (lightInstance != null) {
                lightInstance.detachFromRenderer();
            }
            Collider collider = this.collider;
            if (collider != null) {
                collider.setAttachedCollisionSystem(null);
            }
            onDeactivate();
            Iterator<LifecycleListener> it = this.lifecycleListeners.iterator();
            while (it.hasNext()) {
                it.next().onDeactivated(this);
            }
            return;
        }
        throw new AssertionError("Cannot call deactivate while already inactive.");
    }

    private void destroyLightInstance() {
        LightInstance lightInstance = this.lightInstance;
        if (lightInstance == null) {
            return;
        }
        if (this.active) {
            lightInstance.detachFromRenderer();
        }
        this.lightInstance.dispose();
        this.lightInstance = null;
    }

    private boolean dispatchToViewRenderable(MotionEvent motionEvent) {
        return ViewTouchHelpers.dispatchTouchEventToView(this, motionEvent);
    }

    private void dispatchTransformChanged(Node node) {
        onTransformChange(node);
        for (int i = 0; i < this.transformChangedListeners.size(); i++) {
            this.transformChangedListeners.get(i).onTransformChanged(this, node);
        }
    }

    private Renderer getRendererOrDie() {
        Scene scene = this.scene;
        if (scene != null) {
            return (Renderer) Preconditions.checkNotNull(scene.getView().getRenderer());
        }
        throw new IllegalStateException("Unable to get Renderer.");
    }

    private int getScaledTouchSlop() {
        Scene scene = getScene();
        if (scene == null || !AndroidPreconditions.isAndroidApiAvailable() || AndroidPreconditions.isUnderTesting()) {
            return 8;
        }
        return ViewConfiguration.get(scene.getView().getContext()).getScaledTouchSlop();
    }

    private Matrix getWorldModelMatrixInternal() {
        if ((this.dirtyTransformFlags & 2) == 2) {
            Node node = this.parentAsNode;
            if (node == null) {
                this.cachedWorldModelMatrix.set(getLocalModelMatrixInternal().data);
            } else {
                Matrix.multiply(node.getWorldModelMatrixInternal(), getLocalModelMatrixInternal(), this.cachedWorldModelMatrix);
            }
            this.dirtyTransformFlags &= -3;
        }
        return this.cachedWorldModelMatrix;
    }

    private Vector3 getWorldPositionInternal() {
        if ((this.dirtyTransformFlags & 8) == 8) {
            if (this.parentAsNode != null) {
                getWorldModelMatrixInternal().decomposeTranslation(this.cachedWorldPosition);
            } else {
                this.cachedWorldPosition.set(this.localPosition);
            }
            this.dirtyTransformFlags &= -9;
        }
        return this.cachedWorldPosition;
    }

    private Quaternion getWorldRotationInternal() {
        if ((this.dirtyTransformFlags & 16) == 16) {
            if (this.parentAsNode != null) {
                getWorldModelMatrixInternal().decomposeRotation(getWorldScaleInternal(), this.cachedWorldRotation);
            } else {
                this.cachedWorldRotation.set(this.localRotation);
            }
            this.dirtyTransformFlags &= -17;
        }
        return this.cachedWorldRotation;
    }

    private Vector3 getWorldScaleInternal() {
        if ((this.dirtyTransformFlags & 32) == 32) {
            if (this.parentAsNode != null) {
                getWorldModelMatrixInternal().decomposeScale(this.cachedWorldScale);
            } else {
                this.cachedWorldScale.set(this.localScale);
            }
            this.dirtyTransformFlags &= -33;
        }
        return this.cachedWorldScale;
    }

    private final void markTransformChangedRecursively(int i, Node node) {
        boolean z;
        Collider collider;
        int i2 = this.dirtyTransformFlags;
        boolean z2 = true;
        if ((i2 & i) != i) {
            int i3 = i2 | i;
            this.dirtyTransformFlags = i3;
            if ((i3 & 2) == 2 && (collider = this.collider) != null) {
                collider.markWorldShapeDirty();
            }
            z = true;
        } else {
            z = false;
        }
        if (node.allowDispatchTransformChangedListeners) {
            dispatchTransformChanged(node);
        } else {
            z2 = z;
        }
        if (z2) {
            List<Node> children = getChildren();
            for (int i4 = 0; i4 < children.size(); i4++) {
                children.get(i4).markTransformChangedRecursively(i, node);
            }
        }
    }

    private void refreshCollider() {
        Scene scene;
        CollisionShape collisionShape = this.collisionShape;
        Renderable renderable = getRenderable();
        if (collisionShape == null && renderable != null) {
            collisionShape = renderable.getCollisionShape();
        }
        if (collisionShape != null) {
            Collider collider = this.collider;
            if (collider == null) {
                Collider collider2 = new Collider(this, collisionShape);
                this.collider = collider2;
                if (!this.active || (scene = this.scene) == null) {
                    return;
                }
                collider2.setAttachedCollisionSystem(scene.collisionSystem);
                return;
            } else if (collider.getShape() != collisionShape) {
                this.collider.setShape(collisionShape);
                return;
            } else {
                return;
            }
        }
        Collider collider3 = this.collider;
        if (collider3 != null) {
            collider3.setAttachedCollisionSystem(null);
            this.collider = null;
        }
    }

    private void setSceneRecursivelyInternal(Scene scene) {
        this.scene = scene;
        for (Node node : getChildren()) {
            node.setSceneRecursively(scene);
        }
    }

    private boolean shouldBeActive() {
        if (this.enabled && this.scene != null) {
            Node node = this.parentAsNode;
            return node == null || node.isActive();
        }
        return false;
    }

    private void updateActiveStatusRecursively() {
        boolean shouldBeActive = shouldBeActive();
        if (this.active != shouldBeActive) {
            if (shouldBeActive) {
                activate();
            } else {
                deactivate();
            }
        }
        for (Node node : getChildren()) {
            node.updateActiveStatusRecursively();
        }
    }

    public void addLifecycleListener(LifecycleListener lifecycleListener) {
        if (this.lifecycleListeners.contains(lifecycleListener)) {
            return;
        }
        this.lifecycleListeners.add(lifecycleListener);
    }

    public void addTransformChangedListener(TransformChangedListener transformChangedListener) {
        if (this.transformChangedListeners.contains(transformChangedListener)) {
            return;
        }
        this.transformChangedListeners.add(transformChangedListener);
    }

    @Override // com.google.ar.sceneform.NodeParent
    public void callOnHierarchy(Consumer<Node> consumer) {
        consumer.accept(this);
        super.callOnHierarchy(consumer);
    }

    @Override // com.google.ar.sceneform.NodeParent
    public final boolean canAddChild(Node node, StringBuilder sb) {
        if (super.canAddChild(node, sb)) {
            if (isDescendantOf(node)) {
                sb.append("Cannot add child: A node's parent cannot be one of its descendants.");
                return false;
            }
            return true;
        }
        return false;
    }

    public boolean dispatchTouchEvent(HitTestResult hitTestResult, MotionEvent motionEvent) {
        Preconditions.checkNotNull(hitTestResult, "Parameter \"hitTestResult\" was null.");
        Preconditions.checkNotNull(motionEvent, "Parameter \"motionEvent\" was null.");
        if (isActive()) {
            if (dispatchToViewRenderable(motionEvent)) {
                return true;
            }
            OnTouchListener onTouchListener = this.onTouchListener;
            if (onTouchListener == null || !onTouchListener.onTouch(hitTestResult, motionEvent)) {
                return onTouchEvent(hitTestResult, motionEvent);
            }
            return true;
        }
        return false;
    }

    public final void dispatchUpdate(FrameTime frameTime) {
        if (isActive()) {
            Renderable renderable = getRenderable();
            if (renderable != null && renderable.getId().checkChanged(this.renderableId)) {
                refreshCollider();
                this.renderableId = renderable.getId().get();
            }
            onUpdate(frameTime);
            Iterator<LifecycleListener> it = this.lifecycleListeners.iterator();
            while (it.hasNext()) {
                it.next().onUpdated(this, frameTime);
            }
        }
    }

    @Override // com.google.ar.sceneform.NodeParent
    public Node findInHierarchy(Predicate<Node> predicate) {
        return predicate.test(this) ? this : super.findInHierarchy(predicate);
    }

    public final Vector3 getBack() {
        return localToWorldDirection(Vector3.back());
    }

    public final Collider getCollider() {
        return this.collider;
    }

    public CollisionShape getCollisionShape() {
        Collider collider = this.collider;
        if (collider != null) {
            return collider.getShape();
        }
        return null;
    }

    public final Vector3 getDown() {
        return localToWorldDirection(Vector3.down());
    }

    public final Vector3 getForward() {
        return localToWorldDirection(Vector3.forward());
    }

    public final Vector3 getLeft() {
        return localToWorldDirection(Vector3.left());
    }

    public Light getLight() {
        LightInstance lightInstance = this.lightInstance;
        if (lightInstance != null) {
            return lightInstance.getLight();
        }
        return null;
    }

    public Matrix getLocalModelMatrixInternal() {
        if ((this.dirtyTransformFlags & 1) == 1) {
            this.cachedLocalModelMatrix.makeTrs(this.localPosition, this.localRotation, this.localScale);
            this.dirtyTransformFlags &= -2;
        }
        return this.cachedLocalModelMatrix;
    }

    public final Vector3 getLocalPosition() {
        return new Vector3(this.localPosition);
    }

    public final Quaternion getLocalRotation() {
        return new Quaternion(this.localRotation);
    }

    public final Vector3 getLocalScale() {
        return new Vector3(this.localScale);
    }

    public final String getName() {
        return this.name;
    }

    public int getNameHash() {
        return this.nameHash;
    }

    public final NodeParent getNodeParent() {
        return this.parent;
    }

    public final Node getParent() {
        return this.parentAsNode;
    }

    public Renderable getRenderable() {
        RenderableInstance renderableInstance = this.renderableInstance;
        if (renderableInstance == null) {
            return null;
        }
        return renderableInstance.getRenderable();
    }

    public RenderableInstance getRenderableInstance() {
        return this.renderableInstance;
    }

    public final Vector3 getRight() {
        return localToWorldDirection(Vector3.right());
    }

    public final Scene getScene() {
        return this.scene;
    }

    public final Vector3 getUp() {
        return localToWorldDirection(Vector3.up());
    }

    @Override // com.google.ar.sceneform.common.TransformProvider
    public final Matrix getWorldModelMatrix() {
        return getWorldModelMatrixInternal();
    }

    public Matrix getWorldModelMatrixInverseInternal() {
        if ((this.dirtyTransformFlags & 4) == 4) {
            Matrix.invert(getWorldModelMatrixInternal(), this.cachedWorldModelMatrixInverse);
            this.dirtyTransformFlags &= -5;
        }
        return this.cachedWorldModelMatrixInverse;
    }

    public final Vector3 getWorldPosition() {
        return new Vector3(getWorldPositionInternal());
    }

    public final Quaternion getWorldRotation() {
        return new Quaternion(getWorldRotationInternal());
    }

    public final Vector3 getWorldScale() {
        return new Vector3(getWorldScaleInternal());
    }

    public final boolean isActive() {
        return this.active;
    }

    public final boolean isDescendantOf(NodeParent nodeParent) {
        Preconditions.checkNotNull(nodeParent, "Parameter \"ancestor\" was null.");
        NodeParent nodeParent2 = this.parent;
        Node node = this.parentAsNode;
        while (nodeParent2 != null) {
            if (nodeParent2 == nodeParent) {
                return true;
            }
            if (node == null) {
                return false;
            }
            nodeParent2 = node.parent;
            node = node.parentAsNode;
        }
        return false;
    }

    public final boolean isEnabled() {
        return this.enabled;
    }

    public boolean isTopLevel() {
        NodeParent nodeParent = this.parent;
        return nodeParent == null || nodeParent == this.scene;
    }

    public final Vector3 localToWorldDirection(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"direction\" was null.");
        return Quaternion.rotateVector(getWorldRotationInternal(), vector3);
    }

    public final Vector3 localToWorldPoint(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"point\" was null.");
        return getWorldModelMatrixInternal().transformPoint(vector3);
    }

    public void onActivate() {
    }

    @Override // com.google.ar.sceneform.NodeParent
    public final void onAddChild(Node node) {
        super.onAddChild(node);
        node.parentAsNode = this;
        node.markTransformChangedRecursively(62, node);
        node.setSceneRecursively(this.scene);
    }

    public void onDeactivate() {
    }

    @Override // com.google.ar.sceneform.NodeParent
    public final void onRemoveChild(Node node) {
        super.onRemoveChild(node);
        node.parentAsNode = null;
        node.markTransformChangedRecursively(62, node);
        node.setSceneRecursively(null);
    }

    public boolean onTouchEvent(HitTestResult hitTestResult, MotionEvent motionEvent) {
        Node node;
        TapTrackingData tapTrackingData;
        OnTapListener onTapListener;
        OnAccurateTapListner onAccurateTapListner;
        Preconditions.checkNotNull(hitTestResult, "Parameter \"hitTestResult\" was null.");
        Preconditions.checkNotNull(motionEvent, "Parameter \"motionEvent\" was null.");
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 0 || !isActive()) {
            this.tapTrackingData = null;
        }
        if (actionMasked == 0) {
            this.accurateTouch = 0;
        } else if (actionMasked == 2) {
            this.accurateTouch++;
        }
        if (actionMasked != 0) {
            if ((actionMasked != 1 && actionMasked != 2) || (tapTrackingData = this.tapTrackingData) == null) {
                return false;
            }
            if ((hitTestResult.getNode() == tapTrackingData.downNode) || Vector3.subtract(tapTrackingData.downPosition, new Vector3(motionEvent.getX(), motionEvent.getY(), StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)).length() < getScaledTouchSlop()) {
                if (actionMasked == 1 && this.accurateTouch < 10 && (onAccurateTapListner = this.onAccurateTapListner) != null) {
                    this.accurateTouch = 0;
                    onAccurateTapListner.onAccurateTap(hitTestResult, motionEvent);
                    this.tapTrackingData = null;
                } else if (actionMasked == 1 && (onTapListener = this.onTapListener) != null) {
                    onTapListener.onTap(hitTestResult, motionEvent);
                    this.tapTrackingData = null;
                }
            } else {
                this.tapTrackingData = null;
                return false;
            }
        } else if ((this.onTapListener == null && this.onAccurateTapListner == null) || (node = hitTestResult.getNode()) == null) {
            return false;
        } else {
            this.tapTrackingData = new TapTrackingData(node, new Vector3(motionEvent.getX(), motionEvent.getY(), StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
        }
        return true;
    }

    public void onTransformChange(Node node) {
    }

    public void onUpdate(FrameTime frameTime) {
    }

    public void removeLifecycleListener(LifecycleListener lifecycleListener) {
        this.lifecycleListeners.remove(lifecycleListener);
    }

    public void removeTransformChangedListener(TransformChangedListener transformChangedListener) {
        this.transformChangedListeners.remove(transformChangedListener);
    }

    public void setCollisionShape(CollisionShape collisionShape) {
        AndroidPreconditions.checkUiThread();
        this.collisionShape = collisionShape;
        refreshCollider();
    }

    public final void setEnabled(boolean z) {
        AndroidPreconditions.checkUiThread();
        if (this.enabled == z) {
            return;
        }
        this.enabled = z;
        updateActiveStatusRecursively();
    }

    public void setLight(Light light) {
        if (getLight() == light) {
            return;
        }
        destroyLightInstance();
        if (light != null) {
            createLightInstance(light);
        }
    }

    public void setLocalPosition(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"position\" was null.");
        this.localPosition.set(vector3);
        markTransformChangedRecursively(63, this);
    }

    public void setLocalRotation(Quaternion quaternion) {
        Preconditions.checkNotNull(quaternion, "Parameter \"rotation\" was null.");
        this.localRotation.set(quaternion);
        markTransformChangedRecursively(63, this);
    }

    public void setLocalScale(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"scale\" was null.");
        this.localScale.set(vector3);
        markTransformChangedRecursively(63, this);
    }

    public final void setLookDirection(Vector3 vector3, Vector3 vector32) {
        setWorldRotation(Quaternion.lookRotation(vector3, vector32));
    }

    public final void setName(String str) {
        Preconditions.checkNotNull(str, "Parameter \"name\" was null.");
        this.name = str;
        this.nameHash = str.hashCode();
    }

    public void setOnAccurateTapListner(OnAccurateTapListner onAccurateTapListner) {
        if (onAccurateTapListner != this.onAccurateTapListner) {
            this.tapTrackingData = null;
        }
        this.onAccurateTapListner = onAccurateTapListner;
    }

    public void setOnTapListener(OnTapListener onTapListener) {
        if (onTapListener != this.onTapListener) {
            this.tapTrackingData = null;
        }
        this.onTapListener = onTapListener;
    }

    public void setOnTouchListener(OnTouchListener onTouchListener) {
        this.onTouchListener = onTouchListener;
    }

    public void setParent(NodeParent nodeParent) {
        AndroidPreconditions.checkUiThread();
        NodeParent nodeParent2 = this.parent;
        if (nodeParent == nodeParent2) {
            return;
        }
        this.allowDispatchTransformChangedListeners = false;
        if (nodeParent != null) {
            nodeParent.addChild(this);
        } else if (nodeParent2 != null) {
            nodeParent2.removeChild(this);
        }
        this.allowDispatchTransformChangedListeners = true;
        markTransformChangedRecursively(62, this);
    }

    public RenderableInstance setRenderable(Renderable renderable) {
        Scene scene;
        AndroidPreconditions.checkUiThread();
        RenderableInstance renderableInstance = this.renderableInstance;
        if (renderableInstance != null && renderableInstance.getRenderable() == renderable) {
            return this.renderableInstance;
        }
        RenderableInstance renderableInstance2 = this.renderableInstance;
        if (renderableInstance2 != null) {
            if (this.active) {
                renderableInstance2.detachFromRenderer();
            }
            this.renderableInstance = null;
        }
        if (renderable != null) {
            RenderableInstance createInstance = renderable.createInstance(this);
            if (this.active && (scene = this.scene) != null && !scene.isUnderTesting()) {
                createInstance.attachToRenderer(getRendererOrDie());
            }
            this.renderableInstance = createInstance;
            this.renderableId = renderable.getId().get();
        } else {
            this.renderableId = 0;
        }
        refreshCollider();
        return this.renderableInstance;
    }

    public final void setSceneRecursively(Scene scene) {
        AndroidPreconditions.checkUiThread();
        setSceneRecursivelyInternal(scene);
        updateActiveStatusRecursively();
    }

    public void setWorldPosition(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"position\" was null.");
        Node node = this.parentAsNode;
        if (node == null) {
            this.localPosition.set(vector3);
        } else {
            this.localPosition.set(node.worldToLocalPoint(vector3));
        }
        markTransformChangedRecursively(63, this);
        this.cachedWorldPosition.set(vector3);
        this.dirtyTransformFlags &= -9;
    }

    public void setWorldRotation(Quaternion quaternion) {
        Preconditions.checkNotNull(quaternion, "Parameter \"rotation\" was null.");
        Node node = this.parentAsNode;
        if (node == null) {
            this.localRotation.set(quaternion);
        } else {
            this.localRotation.set(Quaternion.multiply(node.getWorldRotationInternal().inverted(), quaternion));
        }
        markTransformChangedRecursively(63, this);
        this.cachedWorldRotation.set(quaternion);
        this.dirtyTransformFlags &= -17;
    }

    public void setWorldScale(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"scale\" was null.");
        Node node = this.parentAsNode;
        if (node != null) {
            this.allowDispatchTransformChangedListeners = false;
            setLocalScale(Vector3.one());
            this.allowDispatchTransformChangedListeners = true;
            Matrix localModelMatrixInternal = getLocalModelMatrixInternal();
            Matrix.multiply(node.getWorldModelMatrixInternal(), localModelMatrixInternal, this.cachedWorldModelMatrix);
            localModelMatrixInternal.makeScale(vector3);
            Matrix matrix = this.cachedWorldModelMatrix;
            Matrix.invert(matrix, matrix);
            Matrix.multiply(matrix, localModelMatrixInternal, matrix);
            matrix.decomposeScale(this.localScale);
            setLocalScale(this.localScale);
        } else {
            setLocalScale(vector3);
        }
        this.cachedWorldScale.set(vector3);
        this.dirtyTransformFlags &= -33;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.name);
        sb.append("(");
        return a.v(sb, super.toString(), ")");
    }

    public final Vector3 worldToLocalDirection(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"direction\" was null.");
        return Quaternion.inverseRotateVector(getWorldRotationInternal(), vector3);
    }

    public final Vector3 worldToLocalPoint(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"point\" was null.");
        return getWorldModelMatrixInverseInternal().transformPoint(vector3);
    }

    public final void setLookDirection(Vector3 vector3) {
        Vector3 up = Vector3.up();
        if (Math.abs(Vector3.dot(vector3, up)) > DIRECTION_UP_EPSILON) {
            up = new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
        }
        setLookDirection(vector3, up);
    }
}