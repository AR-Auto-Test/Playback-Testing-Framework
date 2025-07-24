package c.e.b;

import com.ibosoninnov.unitear.NonARCoreActivitySceneform;
import java.util.TimerTask;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class he extends TimerTask {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f4837b;

    /* compiled from: NonARCoreActivitySceneform.java */
    /* loaded from: classes2.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            NonARCoreActivitySceneform nonARCoreActivitySceneform = he.this.f4837b;
            if (nonARCoreActivitySceneform.m0) {
                int i = nonARCoreActivitySceneform.V + 1;
                nonARCoreActivitySceneform.V = i;
                if (i >= 60) {
                    nonARCoreActivitySceneform.V = 0;
                    nonARCoreActivitySceneform.W++;
                }
                nonARCoreActivitySceneform.S.setText(String.format("%02d:%02d", Integer.valueOf(nonARCoreActivitySceneform.W), Integer.valueOf(he.this.f4837b.V)));
            }
        }
    }

    public he(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f4837b = nonARCoreActivitySceneform;
    }

    @Override // java.util.TimerTask, java.lang.Runnable
    public void run() {
        this.f4837b.runOnUiThread(new a());
    }
}