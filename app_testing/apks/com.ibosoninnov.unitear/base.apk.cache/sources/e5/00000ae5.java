package c.e.b;

import android.widget.ProgressBar;
import android.widget.TextView;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.ux.SimpleTransformableNode;

/* compiled from: LoaderARContentSceneform.java */
/* loaded from: classes2.dex */
public class qd implements c.e.b.gf.c {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Node f5167a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ve f5168b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ SimpleTransformableNode f5169c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ Node f5170d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ hd f5171e;

    public qd(hd hdVar, Node node, ve veVar, SimpleTransformableNode simpleTransformableNode, Node node2) {
        this.f5171e = hdVar;
        this.f5167a = node;
        this.f5168b = veVar;
        this.f5169c = simpleTransformableNode;
        this.f5170d = node2;
    }

    @Override // c.e.b.gf.c
    public void a(String str, int i, String str2) {
        String str3 = i + " %";
        hd hdVar = this.f5171e;
        ProgressBar progressBar = hdVar.Q;
        if (progressBar != null && i != 100 && hdVar.S != null) {
            progressBar.setProgress(i);
            this.f5171e.S.setText(str3);
            return;
        }
        TextView textView = hdVar.S;
        if (textView != null) {
            textView.setText("");
        }
    }

    @Override // c.e.b.gf.c
    public void b(String str, String str2) {
        if (str2.length() == 0) {
            hd hdVar = this.f5171e;
            int i = hdVar.r - 1;
            hdVar.r = i;
            if (i == 0) {
                hdVar.k();
                return;
            }
            return;
        }
        this.f5171e.t(str2, this.f5167a, this.f5168b, this.f5169c, this.f5170d);
    }
}