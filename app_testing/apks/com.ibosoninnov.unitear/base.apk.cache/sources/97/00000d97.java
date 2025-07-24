package com.google.android.gms.common.api;

import androidx.annotation.RecentlyNonNull;
import com.google.android.gms.common.api.Result;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class BatchResultToken<R extends Result> {
    @RecentlyNonNull
    public final int mId;

    public BatchResultToken(int i) {
        this.mId = i;
    }
}