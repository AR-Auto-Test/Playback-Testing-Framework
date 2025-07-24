package c.e.b;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Scene;
import java.util.Objects;

/* compiled from: LoaderARContentGroundPlaneSceneformARCore.java */
/* loaded from: classes2.dex */
public class yc implements Scene.OnPeekTouchListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ vc f5449b;

    public yc(vc vcVar) {
        this.f5449b = vcVar;
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        Objects.requireNonNull(this.f5449b);
        vc vcVar = this.f5449b;
        if (!vcVar.n) {
            vc.a(vcVar, true);
        } else if (motionEvent.getAction() == 1) {
            vc vcVar2 = this.f5449b;
            vcVar2.p.postDelayed(vcVar2.q, 2000L);
        }
    }
}