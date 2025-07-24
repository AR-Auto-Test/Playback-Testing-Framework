package c.e.b.p000if;

import android.view.MotionEvent;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.ux.SimpleTransformableNode;

/* compiled from: TouchRotateSceneform.java */
/* renamed from: c.e.b.if.p  reason: invalid package */
/* loaded from: classes2.dex */
public class p {

    /* renamed from: a  reason: collision with root package name */
    public float f4899a;

    /* renamed from: b  reason: collision with root package name */
    public Quaternion f4900b;

    /* renamed from: c  reason: collision with root package name */
    public int f4901c;

    public void a(MotionEvent motionEvent, SimpleTransformableNode simpleTransformableNode) {
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 0) {
            this.f4901c = 0;
        } else if (actionMasked == 2 && motionEvent.getPointerCount() == 1) {
            this.f4901c++;
        }
        int action = motionEvent.getAction();
        if (motionEvent.getPointerCount() == motionEvent.getAction()) {
            return;
        }
        if (action == 0) {
            if (simpleTransformableNode != null) {
                this.f4900b = simpleTransformableNode.getLocalRotation();
            }
            this.f4899a = motionEvent.getX();
        }
        if (action != 2 || simpleTransformableNode == null || Math.abs(this.f4899a - motionEvent.getX()) <= 5.0f || Math.abs(this.f4899a - motionEvent.getX()) <= 5.0f || simpleTransformableNode.isTransforming() || this.f4901c <= 5 || this.f4900b == null) {
            return;
        }
        simpleTransformableNode.setLocalRotation(Quaternion.multiply(this.f4900b, Quaternion.eulerAngles(new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, (this.f4899a - motionEvent.getX()) * (-0.2f)))));
    }
}