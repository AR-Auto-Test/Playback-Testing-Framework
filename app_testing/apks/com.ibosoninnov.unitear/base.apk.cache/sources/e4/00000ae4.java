package c.e.b;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Scene;

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class qc implements Scene.OnPeekTouchListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ jc f5166b;

    public qc(jc jcVar) {
        this.f5166b = jcVar;
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        this.f5166b.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
    }
}