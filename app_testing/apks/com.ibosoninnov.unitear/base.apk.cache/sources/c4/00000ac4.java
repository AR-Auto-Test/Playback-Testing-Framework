package c.e.b;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Scene;

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class oc implements Scene.OnPeekTouchListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ jc f5100b;

    public oc(jc jcVar) {
        this.f5100b = jcVar;
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        this.f5100b.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
    }
}