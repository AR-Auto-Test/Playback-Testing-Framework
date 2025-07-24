package com.google.android.material.timepicker;

import android.content.Context;
import android.view.View;
import b.j.j.a;
import b.j.j.x.b;

/* loaded from: classes.dex */
public class ClickActionDelegate extends a {
    private final b.a clickAction;

    public ClickActionDelegate(Context context, int i) {
        this.clickAction = new b.a(16, context.getString(i));
    }

    @Override // b.j.j.a
    public void onInitializeAccessibilityNodeInfo(View view, b bVar) {
        super.onInitializeAccessibilityNodeInfo(view, bVar);
        bVar.a(this.clickAction);
    }
}