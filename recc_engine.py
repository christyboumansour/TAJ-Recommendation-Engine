import pandas as pd
from apyori import apriori
import streamlit as st


# Load the dataset
df = pd.read_csv('Sales1.csv')

# Remove rows with a volume of 0
df = df[df['Volume'] != 0]

df= df[df['Area'] != '-']


# Set the title of the app
st.set_page_config(
    page_title="TAJ Recommendation Engine",
    page_icon="🛒",
    layout="wide"
)

# Define style elements
main_bg_color = "#F4F4F4"
sidebar_bg_color = "#EAEAEA"
header_font = ("Helvetica", 32, "bold")

# Apply style to main content
main_container = st.container()
with main_container:
    st.title("Recommendation Engine")
    
    # Step 1: Choose Supermarket Group
    st.header("Step 1: Choose Supermarket Group")
    selected_group = st.selectbox("Select a Supermarket Group", df['Group'].unique())
    st.write("Selected Supermarket Group:", selected_group)
    
    # Step 2: Choose Food Category
    st.markdown("---")
    st.header("Step 2: Choose Food Category")
    st.write("Selected Supermarket Group:", selected_group)  # Display selected group
    selected_category = st.selectbox("Select a Food Category", df['Category'].unique())
    st.write("Selected Food Category:", selected_category)
    
    # Update selected_description
    descriptions_in_category = df[df['Category'] == selected_category]['ProcessedDescription'].unique()
    selected_description = st.selectbox("Select a Product Description", descriptions_in_category)
    st.write("Selected Description:", selected_description)


# Step 3: Apply Apriori and Display Results
st.markdown("---")
st.header("Step 3: Apply Apriori and Display Results")

# Filter the data based on selected group and description
filtered_data = df[df['Group'] == selected_group]

# Prepare the data for Apriori
transactions = filtered_data.groupby(['Account_Number', 'Invoice_Date'])['ProcessedDescription'].apply(lambda x: frozenset(x.dropna().values.flatten()))


# Apply Apriori algorithm
min_support = 0.01  # Set your desired minimum support value
min_confidence = 0.5  # Set your desired minimum confidence value
rules = apriori(transactions.to_list(), min_support=min_support, min_confidence=min_confidence, min_lift=2)
results = list(rules)

# Create DataFrame from inspection
def inspect(results, selected_description):
    grouped_rules = {}  # To store grouped rules
    for result in results:
        antecedent = selected_description
        consequent = None  # Initialize consequent as None
        for stat in result.ordered_statistics:
            if antecedent in stat.items_base:  # Check if antecedent is in items_base
                consequent = ", ".join(stat.items_add)
                break
        if consequent:
            pair = (antecedent, consequent)
            if pair not in grouped_rules:
                grouped_rules[pair] = result
            else:
                # Compare lifts and keep the rule with the higher lift
                existing_lift = grouped_rules[pair].ordered_statistics[0].lift
                new_lift = result.ordered_statistics[0].lift
                if new_lift > existing_lift:
                    grouped_rules[pair] = result
                
    data = []
    for pair, result in grouped_rules.items():
        antecedent, consequent = pair
        support = result.support
        confidence = result.ordered_statistics[0].confidence
        lift = result.ordered_statistics[0].lift
        data.append((antecedent, consequent, support, confidence, lift))
        
    return data


basket_df = pd.DataFrame(inspect(results, selected_description), columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
# Drop duplicates from the DataFrame
basket_df.drop_duplicates(inplace=True)


# Sort DataFrame by Confidence and Lift
basket_df.sort_values(by=['Confidence', 'Lift'], ascending=[False, False], inplace=True)

# Display a header above the DataFrame with smaller font size
st.markdown("<h3 style='font-size: 18px;'> Basket Items based on selected rules:</h3>", unsafe_allow_html=True)

# Display the top 10 resulting rules
st.dataframe(basket_df.head(10))

if not basket_df.empty:
    st.write(
        f"Based on the analysis, the product(s) '{basket_df.at[0, 'Consequent']}' is recommended to be purchased along with '{selected_description}'.")
    st.write(
        f"You were {basket_df.at[0, 'Lift']:.2f} times more likely to buy '{basket_df.at[0, 'Consequent']}' with '{selected_description}'.")
    st.write(
        f"There is {basket_df.at[0, 'Confidence']:.2f} chance that you would buy these items together")
else:
    st.warning(f"No complementary recommendations available for {selected_group} and {selected_description}")

 
    # Step 4: Purchase History
st.markdown("---")
st.header("Step 4: Purchase History")
st.write("Selected Supermarket Group:", selected_group)  # Display selected group

logo_image= "https://tajgroupholding.com/cdn/shop/files/a5a99ea6-e35e-4c61-9d5c-d73392c89e40_160x160@2x.jpg?v=1670953223"
st.sidebar.image(logo_image, width=75)

# Create a sidebar with filter options for purchase history
st.sidebar.title("Purchase History Filters")

   # Filter purchase history based on selected group
filtered_purchase_history = df[df['Group'] == selected_group]

# Apply filters from the sidebar
selected_dates = st.sidebar.date_input("Select Dates", [pd.to_datetime(filtered_purchase_history['Invoice_Date']).min(), pd.to_datetime(filtered_purchase_history['Invoice_Date']).max()])
selected_areas = st.sidebar.multiselect("Select Areas", filtered_purchase_history['Area'].unique())
selected_categories = st.sidebar.multiselect("Select Categories", filtered_purchase_history['Category'].unique())

# Apply filters to the DataFrame
filtered_purchase_history = filtered_purchase_history[
    (pd.to_datetime(filtered_purchase_history['Invoice_Date']).dt.date >= selected_dates[0]) &
    (pd.to_datetime(filtered_purchase_history['Invoice_Date']).dt.date <= selected_dates[1]) &
    (filtered_purchase_history['Area'].isin(selected_areas)) &
    (filtered_purchase_history['Category'].isin(selected_categories))
]

if filtered_purchase_history.empty:
  st.warning("No data available for the selected filters.")
else:
# Display the filtered purchase history with relevant features
  purchase_history_columns = ['Invoice_Date', 'Account_Number' , 'ProcessedDescription', 'Area', 'Category', 'Volume', 'Price(AED)', 'Value']
  formatted_purchase_history = filtered_purchase_history[purchase_history_columns].copy()

# Format numerical features to show only two decimal places
  formatted_purchase_history['Volume'] = formatted_purchase_history['Volume'].apply(lambda x: f"{x:.2f}")
  formatted_purchase_history['Price(AED)'] = formatted_purchase_history['Price(AED)'].apply(lambda x: f"{x:.2f}")
  formatted_purchase_history['Value'] = formatted_purchase_history['Value'].apply(lambda x: f"{x:.2f}")
  
  st.write("Filtered Purchase History")
  st.dataframe(formatted_purchase_history.style.set_properties(**{'background-color': main_bg_color, 'color': 'black'}))

# Add spacing at the bottom
st.markdown("---")
