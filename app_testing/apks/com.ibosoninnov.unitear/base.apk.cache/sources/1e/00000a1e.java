package c.e.b;

import com.ibosoninnov.unitear.NonARCoreActivitySceneform;
import java.util.Timer;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class ge implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f4787b;

    public ge(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f4787b = nonARCoreActivitySceneform;
    }

    @Override // java.lang.Runnable
    public void run() {
        this.f4787b.S.setVisibility(0);
        NonARCoreActivitySceneform nonARCoreActivitySceneform = this.f4787b;
        nonARCoreActivitySceneform.W = 0;
        nonARCoreActivitySceneform.V = 0;
        nonARCoreActivitySceneform.S.setText(String.format("%02d:%02d", 0, Integer.valueOf(this.f4787b.V)));
        NonARCoreActivitySceneform nonARCoreActivitySceneform2 = this.f4787b;
        if (nonARCoreActivitySceneform2.b0 == null) {
            c.e.b.p000if.q qVar = new c.e.b.p000if.q(nonARCoreActivitySceneform2.x);
            nonARCoreActivitySceneform2.b0 = qVar;
            qVar.f4906e = nonARCoreActivitySceneform2.A.getArSceneView();
            int i = nonARCoreActivitySceneform2.getResources().getConfiguration().orientation;
            if (nonARCoreActivitySceneform2.r0 > 1000.0f) {
                nonARCoreActivitySceneform2.b0.d(6, i);
            } else {
                nonARCoreActivitySceneform2.b0.d(5, i);
            }
        }
        if (!nonARCoreActivitySceneform2.X) {
            boolean b2 = nonARCoreActivitySceneform2.b0.b();
            nonARCoreActivitySceneform2.X = b2;
            if (b2) {
                nonARCoreActivitySceneform2.m0 = true;
                Timer timer = nonARCoreActivitySceneform2.Z;
                if (timer != null) {
                    timer.cancel();
                }
                nonARCoreActivitySceneform2.W = 0;
                nonARCoreActivitySceneform2.V = -1;
                Timer timer2 = new Timer();
                nonARCoreActivitySceneform2.Z = timer2;
                timer2.scheduleAtFixedRate(new he(nonARCoreActivitySceneform2), 0L, 1000L);
                nonARCoreActivitySceneform2.x(true);
                return;
            }
            return;
        }
        nonARCoreActivitySceneform2.X = nonARCoreActivitySceneform2.b0.b();
        nonARCoreActivitySceneform2.x(false);
        nonARCoreActivitySceneform2.V = 0;
        nonARCoreActivitySceneform2.W = 0;
        Timer timer3 = nonARCoreActivitySceneform2.Z;
        if (timer3 != null) {
            timer3.cancel();
        }
        String path = nonARCoreActivitySceneform2.b0.i.getPath();
        nonARCoreActivitySceneform2.Y = path;
        nonARCoreActivitySceneform2.z(path, true);
    }
}