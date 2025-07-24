package c.e.b;

import com.google.ar.sceneform.Node;
import java.util.Objects;

/* compiled from: LoaderARContentSceneformARCore.java */
/* loaded from: classes2.dex */
public class xd implements c.e.b.gf.c {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Node f5415a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ve f5416b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Node f5417c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ yd f5418d;

    public xd(yd ydVar, Node node, ve veVar, Node node2) {
        this.f5418d = ydVar;
        this.f5415a = node;
        this.f5416b = veVar;
        this.f5417c = node2;
    }

    @Override // c.e.b.gf.c
    public void a(String str, int i, String str2) {
        Objects.requireNonNull(this.f5418d);
    }

    @Override // c.e.b.gf.c
    public void b(String str, String str2) {
        if (str2.length() == 0) {
            yd ydVar = this.f5418d;
            int i = ydVar.s - 1;
            ydVar.s = i;
            if (i == 0) {
                ydVar.j();
                return;
            }
            return;
        }
        this.f5418d.t(str2, this.f5415a, this.f5416b, this.f5417c);
    }
}