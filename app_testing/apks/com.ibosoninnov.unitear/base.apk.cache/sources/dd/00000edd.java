package com.google.android.gms.common.internal;

import android.content.Context;
import android.content.res.ColorStateList;
import android.content.res.Resources;
import android.graphics.PorterDuff;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.widget.Button;
import c.b.a.a.a;
import com.google.android.gms.base.R;
import com.google.android.gms.common.util.DeviceProperties;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zay extends Button {
    public zay(Context context) {
        this(context, null);
    }

    public final void zaa(Resources resources, int i, int i2) {
        setTypeface(Typeface.DEFAULT_BOLD);
        setTextSize(14.0f);
        int i3 = (int) ((resources.getDisplayMetrics().density * 48.0f) + 0.5f);
        setMinHeight(i3);
        setMinWidth(i3);
        int i4 = R.drawable.common_google_signin_btn_icon_dark;
        int i5 = R.drawable.common_google_signin_btn_icon_light;
        int zaa = zaa(i2, i4, i5, i5);
        int i6 = R.drawable.common_google_signin_btn_text_dark;
        int i7 = R.drawable.common_google_signin_btn_text_light;
        int zaa2 = zaa(i2, i6, i7, i7);
        if (i == 0 || i == 1) {
            zaa = zaa2;
        } else if (i != 2) {
            throw new IllegalStateException(a.g(32, "Unknown button size: ", i));
        }
        Drawable drawable = resources.getDrawable(zaa);
        drawable.setTintList(resources.getColorStateList(R.color.common_google_signin_btn_tint));
        drawable.setTintMode(PorterDuff.Mode.SRC_ATOP);
        setBackgroundDrawable(drawable);
        int i8 = R.color.common_google_signin_btn_text_dark;
        int i9 = R.color.common_google_signin_btn_text_light;
        setTextColor((ColorStateList) Preconditions.checkNotNull(resources.getColorStateList(zaa(i2, i8, i9, i9))));
        if (i == 0) {
            setText(resources.getString(R.string.common_signin_button_text));
        } else if (i == 1) {
            setText(resources.getString(R.string.common_signin_button_text_long));
        } else if (i == 2) {
            setText((CharSequence) null);
        } else {
            throw new IllegalStateException(a.g(32, "Unknown button size: ", i));
        }
        setTransformationMethod(null);
        if (DeviceProperties.isWearable(getContext())) {
            setGravity(19);
        }
    }

    private zay(Context context, AttributeSet attributeSet) {
        super(context, null, 16842824);
    }

    private static int zaa(int i, int i2, int i3, int i4) {
        if (i != 0) {
            if (i != 1) {
                if (i == 2) {
                    return i4;
                }
                throw new IllegalStateException(a.g(33, "Unknown color scheme: ", i));
            }
            return i3;
        }
        return i2;
    }
}