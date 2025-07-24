package b.j.j.x;

import android.os.Bundle;
import android.text.style.ClickableSpan;
import android.view.View;

/* compiled from: AccessibilityClickableSpanCompat.java */
/* loaded from: classes.dex */
public final class a extends ClickableSpan {

    /* renamed from: b  reason: collision with root package name */
    public final int f2255b;

    /* renamed from: c  reason: collision with root package name */
    public final b f2256c;

    /* renamed from: d  reason: collision with root package name */
    public final int f2257d;

    public a(int i, b bVar, int i2) {
        this.f2255b = i;
        this.f2256c = bVar;
        this.f2257d = i2;
    }

    @Override // android.text.style.ClickableSpan
    public void onClick(View view) {
        Bundle bundle = new Bundle();
        bundle.putInt("ACCESSIBILITY_CLICKABLE_SPAN_ID", this.f2255b);
        b bVar = this.f2256c;
        bVar.f2259b.performAction(this.f2257d, bundle);
    }
}