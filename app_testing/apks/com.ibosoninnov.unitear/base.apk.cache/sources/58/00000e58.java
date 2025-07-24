package com.google.android.gms.common.data;

import androidx.annotation.RecentlyNonNull;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public interface DataBufferObserver {

    /* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
    /* loaded from: classes.dex */
    public interface Observable {
        void addObserver(@RecentlyNonNull DataBufferObserver dataBufferObserver);

        void removeObserver(@RecentlyNonNull DataBufferObserver dataBufferObserver);
    }

    void onDataChanged();

    void onDataRangeChanged(@RecentlyNonNull int i, @RecentlyNonNull int i2);

    void onDataRangeInserted(@RecentlyNonNull int i, @RecentlyNonNull int i2);

    void onDataRangeMoved(@RecentlyNonNull int i, @RecentlyNonNull int i2, @RecentlyNonNull int i3);

    void onDataRangeRemoved(@RecentlyNonNull int i, @RecentlyNonNull int i2);
}