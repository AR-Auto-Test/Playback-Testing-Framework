package c.e.b;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Scene;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class cd implements Scene.OnPeekTouchListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ vc f4617b;

    public cd(vc vcVar) {
        this.f4617b = vcVar;
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        this.f4617b.l.getTransformationSystem().onTouch(hitTestResult, motionEvent);
    }
}