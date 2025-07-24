package c.e.b;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Scene;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class ad implements Scene.OnPeekTouchListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ vc f4553b;

    public ad(vc vcVar) {
        this.f4553b = vcVar;
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        this.f4553b.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
    }
}