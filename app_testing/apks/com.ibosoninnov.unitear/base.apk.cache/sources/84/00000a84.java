package c.e.b;

import android.view.View;

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class kc implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f4976b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ String f4977c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ jc f4978d;

    public kc(jc jcVar, int i, String str) {
        this.f4978d = jcVar;
        this.f4976b = i;
        this.f4977c = str;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        jc.c(this.f4978d, this.f4976b, this.f4977c);
    }
}