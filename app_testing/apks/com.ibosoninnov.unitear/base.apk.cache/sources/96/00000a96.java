package c.e.b;

import com.ibosoninnov.unitear.NonARCoreActivitySceneform;

/* compiled from: NonARCoreActivitySceneform.java */
/* loaded from: classes2.dex */
public class le implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ NonARCoreActivitySceneform f5011b;

    public le(NonARCoreActivitySceneform nonARCoreActivitySceneform) {
        this.f5011b = nonARCoreActivitySceneform;
    }

    @Override // java.lang.Runnable
    public void run() {
        this.f5011b.q0.resetAnchor(0.5f, 0.5f);
    }
}