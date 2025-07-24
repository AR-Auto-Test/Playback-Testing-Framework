package com.google.android.material.tabs;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.View;
import b.b.d.a.a;
import com.google.android.material.R;

/* loaded from: classes.dex */
public class TabItem extends View {
    public final int customLayout;
    public final Drawable icon;
    public final CharSequence text;

    public TabItem(Context context) {
        this(context, null);
    }

    public TabItem(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        Drawable drawable;
        int resourceId;
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, R.styleable.TabItem);
        this.text = obtainStyledAttributes.getText(R.styleable.TabItem_android_text);
        int i = R.styleable.TabItem_android_icon;
        if (obtainStyledAttributes.hasValue(i) && (resourceId = obtainStyledAttributes.getResourceId(i, 0)) != 0) {
            drawable = a.a(context, resourceId);
        } else {
            drawable = obtainStyledAttributes.getDrawable(i);
        }
        this.icon = drawable;
        this.customLayout = obtainStyledAttributes.getResourceId(R.styleable.TabItem_android_layout, 0);
        obtainStyledAttributes.recycle();
    }
}