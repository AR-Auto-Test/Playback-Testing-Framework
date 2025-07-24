package c.e.b;

import android.view.MotionEvent;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Scene;
import java.util.Objects;

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class mc implements Scene.OnPeekTouchListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ jc f5040b;

    public mc(jc jcVar) {
        this.f5040b = jcVar;
    }

    @Override // com.google.ar.sceneform.Scene.OnPeekTouchListener
    public void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
        Objects.requireNonNull(this.f5040b);
        jc jcVar = this.f5040b;
        if (!jcVar.n) {
            jc.a(jcVar, true);
        } else if (motionEvent.getAction() == 1) {
            jc jcVar2 = this.f5040b;
            jcVar2.p.postDelayed(jcVar2.q, 2000L);
        }
    }
}