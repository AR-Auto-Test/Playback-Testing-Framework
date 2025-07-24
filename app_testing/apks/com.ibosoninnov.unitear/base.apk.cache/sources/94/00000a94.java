package c.e.b;

import android.view.View;

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class lc implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f5007b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ String f5008c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ jc f5009d;

    public lc(jc jcVar, int i, String str) {
        this.f5009d = jcVar;
        this.f5007b = i;
        this.f5008c = str;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        jc.c(this.f5009d, this.f5007b, this.f5008c);
    }
}