package c.e.b;

/* compiled from: LoaderARContentSceneform.java */
/* loaded from: classes2.dex */
public class ld implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ hd f5010b;

    public ld(hd hdVar) {
        this.f5010b = hdVar;
    }

    @Override // java.lang.Runnable
    public void run() {
        hd hdVar = this.f5010b;
        int i = hdVar.r - 1;
        hdVar.r = i;
        if (i == 0) {
            hdVar.k();
        }
    }
}