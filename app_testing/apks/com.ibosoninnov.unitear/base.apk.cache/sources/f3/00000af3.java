package c.e.b;

import com.ibosoninnov.unitear.ARCoreSceneformActivity;
import java.util.TimerTask;

/* compiled from: ARCoreSceneformActivity.java */
/* loaded from: classes2.dex */
public class rb extends TimerTask {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ARCoreSceneformActivity f5197b;

    public rb(ARCoreSceneformActivity aRCoreSceneformActivity) {
        this.f5197b = aRCoreSceneformActivity;
    }

    @Override // java.util.TimerTask, java.lang.Runnable
    public void run() {
        this.f5197b.runOnUiThread(new Runnable() { // from class: c.e.b.a
            @Override // java.lang.Runnable
            public final void run() {
                rb rbVar = rb.this;
                ARCoreSceneformActivity aRCoreSceneformActivity = rbVar.f5197b;
                if (aRCoreSceneformActivity.F) {
                    int i = aRCoreSceneformActivity.f0 + 1;
                    aRCoreSceneformActivity.f0 = i;
                    if (i >= 60) {
                        aRCoreSceneformActivity.f0 = 0;
                        aRCoreSceneformActivity.g0++;
                    }
                    aRCoreSceneformActivity.a0.setText(String.format("%02d:%02d", Integer.valueOf(aRCoreSceneformActivity.g0), Integer.valueOf(rbVar.f5197b.f0)));
                }
            }
        });
    }
}