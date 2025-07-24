package com.google.ar.sceneform.ux;

import android.util.DisplayMetrics;
import android.util.TypedValue;
import android.view.MotionEvent;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.math.Vector3;
import java.util.HashSet;

/* loaded from: classes.dex */
public class GesturePointersUtility {
    private final DisplayMetrics displayMetrics;
    private final HashSet<Integer> retainedPointerIds = new HashSet<>();

    public GesturePointersUtility(DisplayMetrics displayMetrics) {
        this.displayMetrics = displayMetrics;
    }

    public static Vector3 motionEventToPosition(MotionEvent motionEvent, int i) {
        int findPointerIndex = motionEvent.findPointerIndex(i);
        return new Vector3(motionEvent.getX(findPointerIndex), motionEvent.getY(findPointerIndex), StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    public float inchesToPixels(float f2) {
        return TypedValue.applyDimension(4, f2, this.displayMetrics);
    }

    public boolean isPointerIdRetained(int i) {
        return this.retainedPointerIds.contains(Integer.valueOf(i));
    }

    public float pixelsToInches(float f2) {
        return f2 / TypedValue.applyDimension(4, 1.0f, this.displayMetrics);
    }

    public void releasePointerId(int i) {
        this.retainedPointerIds.remove(Integer.valueOf(i));
    }

    public void retainPointerId(int i) {
        if (isPointerIdRetained(i)) {
            return;
        }
        this.retainedPointerIds.add(Integer.valueOf(i));
    }
}