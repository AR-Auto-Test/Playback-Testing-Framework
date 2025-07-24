package com.google.ar.sceneform;

import android.view.MotionEvent;
import android.view.View;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.collision.Plane;
import com.google.ar.sceneform.collision.Ray;
import com.google.ar.sceneform.collision.RayHit;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class ViewTouchHelpers {

    /* renamed from: com.google.ar.sceneform.ViewTouchHelpers$1  reason: invalid class name */
    /* loaded from: classes.dex */
    public static /* synthetic */ class AnonymousClass1 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$HorizontalAlignment;
        public static final /* synthetic */ int[] $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$VerticalAlignment;

        static {
            ViewRenderable.HorizontalAlignment.values();
            int[] iArr = new int[3];
            $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$HorizontalAlignment = iArr;
            try {
                iArr[ViewRenderable.HorizontalAlignment.LEFT.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$HorizontalAlignment[ViewRenderable.HorizontalAlignment.CENTER.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$HorizontalAlignment[ViewRenderable.HorizontalAlignment.RIGHT.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
            ViewRenderable.VerticalAlignment.values();
            int[] iArr2 = new int[3];
            $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$VerticalAlignment = iArr2;
            try {
                iArr2[ViewRenderable.VerticalAlignment.BOTTOM.ordinal()] = 1;
            } catch (NoSuchFieldError unused4) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$VerticalAlignment[ViewRenderable.VerticalAlignment.CENTER.ordinal()] = 2;
            } catch (NoSuchFieldError unused5) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$ViewRenderable$VerticalAlignment[ViewRenderable.VerticalAlignment.TOP.ordinal()] = 3;
            } catch (NoSuchFieldError unused6) {
            }
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:12:0x004f  */
    /* JADX WARN: Removed duplicated region for block: B:15:0x0054  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static Vector3 convertWorldPositionToLocalView(Node node, Vector3 vector3, ViewRenderable viewRenderable) {
        int ordinal;
        Preconditions.checkNotNull(node, "Parameter \"node\" was null.");
        Preconditions.checkNotNull(vector3, "Parameter \"worldPos\" was null.");
        Preconditions.checkNotNull(viewRenderable, "Parameter \"viewRenderable\" was null.");
        Vector3 worldToLocalPoint = node.worldToLocalPoint(vector3);
        View view = viewRenderable.getView();
        int width = view.getWidth();
        int height = view.getHeight();
        float pixelsToMetersRatio = getPixelsToMetersRatio(viewRenderable);
        int i = (int) (worldToLocalPoint.x * pixelsToMetersRatio);
        int i2 = (int) (worldToLocalPoint.y * pixelsToMetersRatio);
        int i3 = width / 2;
        int i4 = height / 2;
        int ordinal2 = viewRenderable.getVerticalAlignment().ordinal();
        if (ordinal2 != 0) {
            if (ordinal2 != 1) {
                if (ordinal2 == 2) {
                    i2 += height;
                }
                ordinal = viewRenderable.getHorizontalAlignment().ordinal();
                if (ordinal != 1) {
                    i += i3;
                } else if (ordinal == 2) {
                    i += width;
                }
                return new Vector3(i, i2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            }
            i2 += i4;
        }
        i2 = height - i2;
        ordinal = viewRenderable.getHorizontalAlignment().ordinal();
        if (ordinal != 1) {
        }
        return new Vector3(i, i2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    public static boolean dispatchTouchEventToView(Node node, MotionEvent motionEvent) {
        Scene scene;
        ViewRenderable viewRenderable;
        Preconditions.checkNotNull(node, "Parameter \"node\" was null.");
        Preconditions.checkNotNull(motionEvent, "Parameter \"motionEvent\" was null.");
        if ((node.getRenderable() instanceof ViewRenderable) && node.isActive() && (scene = node.getScene()) != null && (viewRenderable = (ViewRenderable) node.getRenderable()) != null) {
            int pointerCount = motionEvent.getPointerCount();
            MotionEvent.PointerProperties[] pointerPropertiesArr = new MotionEvent.PointerProperties[pointerCount];
            MotionEvent.PointerCoords[] pointerCoordsArr = new MotionEvent.PointerCoords[pointerCount];
            Plane plane = new Plane(node.getWorldPosition(), node.getForward());
            RayHit rayHit = new RayHit();
            Plane plane2 = new Plane(node.getWorldPosition(), node.getBack());
            for (int i = 0; i < pointerCount; i++) {
                MotionEvent.PointerProperties pointerProperties = new MotionEvent.PointerProperties();
                MotionEvent.PointerCoords pointerCoords = new MotionEvent.PointerCoords();
                motionEvent.getPointerProperties(i, pointerProperties);
                motionEvent.getPointerCoords(i, pointerCoords);
                Ray screenPointToRay = scene.getCamera().screenPointToRay(pointerCoords.x, pointerCoords.y);
                if (plane.rayIntersection(screenPointToRay, rayHit)) {
                    Vector3 convertWorldPositionToLocalView = convertWorldPositionToLocalView(node, rayHit.getPoint(), viewRenderable);
                    pointerCoords.x = convertWorldPositionToLocalView.x;
                    pointerCoords.y = convertWorldPositionToLocalView.y;
                } else if (plane2.rayIntersection(screenPointToRay, rayHit)) {
                    Vector3 convertWorldPositionToLocalView2 = convertWorldPositionToLocalView(node, rayHit.getPoint(), viewRenderable);
                    pointerCoords.x = viewRenderable.getView().getWidth() - convertWorldPositionToLocalView2.x;
                    pointerCoords.y = convertWorldPositionToLocalView2.y;
                } else {
                    pointerCoords.clear();
                    pointerProperties.clear();
                }
                pointerPropertiesArr[i] = pointerProperties;
                pointerCoordsArr[i] = pointerCoords;
            }
            return viewRenderable.getView().dispatchTouchEvent(MotionEvent.obtain(motionEvent.getDownTime(), motionEvent.getEventTime(), motionEvent.getAction(), pointerCount, pointerPropertiesArr, pointerCoordsArr, motionEvent.getMetaState(), motionEvent.getButtonState(), motionEvent.getXPrecision(), motionEvent.getYPrecision(), motionEvent.getDeviceId(), motionEvent.getEdgeFlags(), motionEvent.getSource(), motionEvent.getFlags()));
        }
        return false;
    }

    private static float getPixelsToMetersRatio(ViewRenderable viewRenderable) {
        int width = viewRenderable.getView().getWidth();
        float f2 = viewRenderable.getSizer().getSize(viewRenderable.getView()).x;
        return f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD : width / f2;
    }
}